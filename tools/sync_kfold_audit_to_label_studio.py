"""Idempotently update K-fold audit hints in an existing Label Studio project."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from car5_autolabel.integrations.label_studio_kfold_sync import (
    candidate_to_label_studio_result,
    generate_audit_candidates,
    replace_audit_results,
    update_label_config,
    update_task_audit_data,
)
from car5_autolabel.integrations.label_studio_project_api import (
    LabelStudioProjectApi,
    LabelStudioProjectSnapshot,
    snapshot_to_sync_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        help="Optional SQLite database. Omit to load state through the Label Studio API.",
    )
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--report", type=Path, help="Write the final sync report as JSON")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent API writes; keep 1 for Label Studio installations backed by SQLite",
    )
    parser.add_argument("--operating-score-threshold", type=float, default=0.2)
    parser.add_argument(
        "--possible-missing-score-threshold",
        type=float,
        default=0.4,
        help=(
            "Minimum score for an unmatched prediction to become a possible-missing "
            "hint; matching still uses --operating-score-threshold"
        ),
    )
    parser.add_argument("--minimum-long-side-px", type=int, default=70)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.7)
    parser.add_argument("--cross-class-nms-iou-threshold", type=float, default=0.95)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates. Without this flag the command only prints a dry-run report.",
    )
    parser.add_argument(
        "--reuse-backup",
        action="store_true",
        help="Reuse an existing integrity-checked backup after an interrupted update",
    )
    parser.add_argument(
        "--direct-sqlite-update",
        action="store_true",
        help=(
            "Update existing JSON rows in one SQLite transaction. Stop Label Studio "
            "before using this mode."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_state(database: Path, project_id: int) -> dict[str, Any]:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        project = connection.execute(
            "SELECT label_config, created_by_id FROM project WHERE id = ?", (project_id,)
        ).fetchone()
        if project is None:
            raise ValueError(f"project {project_id} does not exist")
        token = connection.execute(
            "SELECT key FROM authtoken_token WHERE user_id = ? ORDER BY created DESC LIMIT 1",
            (project[1],),
        ).fetchone()
        if token is None:
            raise RuntimeError(f"project owner {project[1]} has no API token")
        tasks = {
            int(task_id): json.loads(raw_data)
            for task_id, raw_data in connection.execute(
                "SELECT id, data FROM task WHERE project_id = ? ORDER BY id", (project_id,)
            )
        }
        annotations: dict[int, list[tuple[int, list[dict[str, Any]]]]] = {}
        for annotation_id, task_id, raw_results in connection.execute(
            """
            SELECT id, task_id, result FROM task_completion
            WHERE project_id = ? AND was_cancelled = 0 ORDER BY id
            """,
            (project_id,),
        ):
            annotations.setdefault(int(task_id), []).append(
                (int(annotation_id), json.loads(raw_results or "[]"))
            )
        drafts: dict[int, list[tuple[int, list[dict[str, Any]]]]] = {}
        for draft_id, task_id, raw_results in connection.execute(
            """
            SELECT draft.id, draft.task_id, draft.result
            FROM tasks_annotationdraft AS draft
            JOIN task ON task.id = draft.task_id
            WHERE task.project_id = ? ORDER BY draft.id
            """,
            (project_id,),
        ):
            drafts.setdefault(int(task_id), []).append(
                (int(draft_id), json.loads(raw_results or "[]"))
            )
        return {
            "token": token[0],
            "label_config": project[0],
            "tasks": tasks,
            "annotations": annotations,
            "drafts": drafts,
        }
    finally:
        connection.close()


def _load_api_state(
    *, base_url: str, token: str, project_id: int
) -> tuple[dict[str, Any], LabelStudioProjectSnapshot]:
    snapshot = LabelStudioProjectApi(base_url=base_url, token=token).snapshot(project_id)
    state = snapshot_to_sync_state(snapshot)
    state["token"] = token
    return state, snapshot


def _backup_api_snapshot(
    snapshot: LabelStudioProjectSnapshot,
    destination: Path,
    *,
    reuse_existing: bool,
) -> None:
    if destination.exists():
        if not reuse_existing:
            raise FileExistsError(f"backup already exists: {destination}")
        existing = _load_json(destination)
        if not isinstance(existing, dict) or not isinstance(existing.get("tasks"), list):
            raise ValueError(f"existing API backup is invalid: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"project": snapshot.project, "tasks": snapshot.tasks},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _backup_database(source: Path, destination: Path, *, reuse_existing: bool) -> None:
    if destination.exists():
        if not reuse_existing:
            raise FileExistsError(f"backup already exists: {destination}")
        connection = sqlite3.connect(destination.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            connection.close()
        if integrity != "ok":
            raise RuntimeError(f"existing backup integrity check failed: {integrity}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        integrity = destination_connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"backup integrity check failed: {integrity}")
    finally:
        destination_connection.close()
        source_connection.close()


def _request(
    *, url: str, token: str, method: str, payload: dict[str, Any]
) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with opener.open(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Label Studio API {method} {url} failed: {error.code} {detail}"
        ) from error
    return json.loads(raw) if raw else {}


def _run_updates(
    updates: list[tuple[str, int, dict[str, Any]]],
    *,
    base_url: str,
    token: str,
    workers: int,
) -> None:
    def update(item: tuple[str, int, dict[str, Any]]) -> None:
        resource, object_id, payload = item
        _request(
            url=f"{base_url}/api/{resource}/{object_id}",
            token=token,
            method="PATCH",
            payload=payload,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(update, item) for item in updates]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            if index % 100 == 0 or index == len(updates):
                print(f"updates={index}/{len(updates)}", flush=True)


def _direct_sqlite_update(
    *,
    database: Path,
    project_id: int,
    updates: list[tuple[str, int, dict[str, Any]]],
    label_config: str,
) -> None:
    """Apply only UPDATE statements atomically; never insert project records."""

    connection = sqlite3.connect(database, timeout=30)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for resource, object_id, payload in updates:
            if resource == "tasks":
                cursor = connection.execute(
                    "UPDATE task SET data = ? WHERE id = ? AND project_id = ?",
                    (
                        json.dumps(payload["data"], ensure_ascii=False, separators=(",", ":")),
                        object_id,
                        project_id,
                    ),
                )
            elif resource == "annotations":
                results = payload["result"]
                result_count = sum(
                    result.get("type") == "rectanglelabels" for result in results
                )
                cursor = connection.execute(
                    """
                    UPDATE task_completion SET result = ?, result_count = ?
                    WHERE id = ? AND project_id = ?
                    """,
                    (
                        json.dumps(results, ensure_ascii=False, separators=(",", ":")),
                        result_count,
                        object_id,
                        project_id,
                    ),
                )
            elif resource == "drafts":
                cursor = connection.execute(
                    """
                    UPDATE tasks_annotationdraft SET result = ?
                    WHERE id = ? AND task_id IN (
                        SELECT id FROM task WHERE project_id = ?
                    )
                    """,
                    (
                        json.dumps(
                            payload["result"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        object_id,
                        project_id,
                    ),
                )
            else:
                raise ValueError(f"unsupported direct-update resource: {resource}")
            if cursor.rowcount != 1:
                raise RuntimeError(f"{resource} {object_id} was not updated exactly once")
        project_cursor = connection.execute(
            "UPDATE project SET label_config = ? WHERE id = ?",
            (label_config, project_id),
        )
        if project_cursor.rowcount != 1:
            raise RuntimeError(f"project {project_id} was not updated exactly once")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"database integrity check failed: {integrity}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def build_updates(
    *,
    state: dict[str, Any],
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    model_version: str,
    operating_score_threshold: float,
    possible_missing_score_threshold: float,
    minimum_long_side_px: int,
    nms_iou_threshold: float,
    cross_class_nms_iou_threshold: float,
) -> tuple[list[tuple[str, int, dict[str, Any]]], dict[str, Any]]:
    candidates, generation_report = generate_audit_candidates(
        ground_truth,
        predictions,
        operating_score_threshold=operating_score_threshold,
        possible_missing_score_threshold=possible_missing_score_threshold,
        minimum_long_side_px=minimum_long_side_px,
        nms_iou_threshold=nms_iou_threshold,
        cross_class_nms_iou_threshold=cross_class_nms_iou_threshold,
    )
    image_by_id = {int(image["id"]): image for image in ground_truth["images"]}
    task_by_image: dict[int, int] = {}
    for image_id, image in image_by_id.items():
        if image.get("label_studio_task_id") is None:
            raise ValueError(f"OOF image {image_id} has no label_studio_task_id")
        task_id = int(image["label_studio_task_id"])
        if task_id in task_by_image.values():
            raise ValueError(f"multiple OOF images map to Label Studio task {task_id}")
        if task_id not in state["tasks"]:
            raise ValueError(f"OOF image {image_id} maps outside the target project: {task_id}")
        task_by_image[image_id] = task_id
    candidates_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_task[task_by_image[int(candidate["image_id"])]].append(candidate)

    hints_by_task: dict[int, list[dict[str, Any]]] = {}
    for image_id, task_id in task_by_image.items():
        image = image_by_id[image_id]
        hints_by_task[task_id] = [
            candidate_to_label_studio_result(
                candidate,
                image_width=int(image["width"]),
                image_height=int(image["height"]),
                model_version=model_version,
            )
            for candidate in candidates_by_task.get(task_id, [])
        ]

    updates: list[tuple[str, int, dict[str, Any]]] = []
    changed_tasks = changed_annotations = changed_drafts = 0
    for task_id, data in state["tasks"].items():
        audited = task_id in hints_by_task
        task_candidates = candidates_by_task.get(task_id, []) if audited else []
        updated_data = update_task_audit_data(
            data, task_candidates, audited=audited, model_version=model_version
        )
        if updated_data != data:
            updates.append(("tasks", task_id, {"data": updated_data}))
            changed_tasks += 1
        new_hints = hints_by_task.get(task_id, [])
        for annotation_id, results in state["annotations"].get(task_id, []):
            updated_results = replace_audit_results(results, new_hints)
            if updated_results != results:
                updates.append(("annotations", annotation_id, {"result": updated_results}))
                changed_annotations += 1
        for draft_id, results in state["drafts"].get(task_id, []):
            updated_results = replace_audit_results(results, new_hints)
            if updated_results != results:
                updates.append(("drafts", draft_id, {"result": updated_results}))
                changed_drafts += 1
    report = {
        **asdict(generation_report),
        "project_tasks": len(state["tasks"]),
        "audited_tasks": len(task_by_image),
        "excluded_tasks": len(state["tasks"]) - len(task_by_image),
        "candidate_tasks": sum(bool(rows) for rows in candidates_by_task.values()),
        "tasks_without_annotations": sum(
            task_id not in state["annotations"] for task_id in state["tasks"]
        ),
        "changed_tasks": changed_tasks,
        "changed_annotations": changed_annotations,
        "changed_drafts": changed_drafts,
        "created_tasks": 0,
        "created_annotations": 0,
    }
    return updates, report


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.direct_sqlite_update and args.database is None:
        raise ValueError("--direct-sqlite-update requires --database")
    combined = args.artifact_root / "oof_combined"
    ground_truth = _load_json(combined / "ground_truth.coco.json")
    predictions = _load_json(combined / "predictions_70px.coco.json")
    if not isinstance(ground_truth, dict) or not isinstance(predictions, list):
        raise ValueError("unexpected OOF artifact format")
    base_url = args.url.rstrip("/")
    api_snapshot: LabelStudioProjectSnapshot | None = None
    if args.database is not None:
        state = _load_state(args.database, args.project_id)
    else:
        token = os.environ.get("CAR5_LABEL_STUDIO_API_TOKEN", "")
        if not token:
            raise ValueError(
                "CAR5_LABEL_STUDIO_API_TOKEN is required when --database is omitted"
            )
        state, api_snapshot = _load_api_state(
            base_url=base_url,
            token=token,
            project_id=args.project_id,
        )
    updates, report = build_updates(
        state=state,
        ground_truth=ground_truth,
        predictions=predictions,
        model_version=args.model_version,
        operating_score_threshold=args.operating_score_threshold,
        possible_missing_score_threshold=args.possible_missing_score_threshold,
        minimum_long_side_px=args.minimum_long_side_px,
        nms_iou_threshold=args.nms_iou_threshold,
        cross_class_nms_iou_threshold=args.cross_class_nms_iou_threshold,
    )
    transitional_label_config = update_label_config(
        state["label_config"], remove_legacy_controls=False
    )
    final_label_config = update_label_config(state["label_config"])
    report["label_config_changed"] = final_label_config != state["label_config"]
    report["dry_run"] = not args.apply
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        _write_report(args.report, report)
        return
    if args.backup is None:
        raise ValueError("--backup is required with --apply")
    if args.database is not None:
        _backup_database(
            args.database, args.backup, reuse_existing=args.reuse_backup
        )
    else:
        assert api_snapshot is not None
        _backup_api_snapshot(
            api_snapshot,
            args.backup,
            reuse_existing=args.reuse_backup,
        )
    if args.direct_sqlite_update:
        assert args.database is not None
        _direct_sqlite_update(
            database=args.database,
            project_id=args.project_id,
            updates=updates,
            label_config=final_label_config,
        )
        print(f"updates={len(updates)}/{len(updates)}", flush=True)
        print(f"backup={args.backup}", flush=True)
        report["applied"] = True
        _write_report(args.report, report)
        return
    # Add the new control before results reference it. Keep legacy controls until
    # their stored results are removed, because Label Studio validates config changes.
    if transitional_label_config != state["label_config"]:
        _request(
            url=f"{base_url}/api/projects/{args.project_id}",
            token=state["token"],
            method="PATCH",
            payload={"label_config": transitional_label_config},
        )
    _run_updates(
        updates,
        base_url=base_url,
        token=state["token"],
        workers=args.workers,
    )
    if final_label_config != transitional_label_config:
        _request(
            url=f"{base_url}/api/projects/{args.project_id}",
            token=state["token"],
            method="PATCH",
            payload={"label_config": final_label_config},
        )
    print(f"backup={args.backup}", flush=True)
    report["applied"] = True
    _write_report(args.report, report)


if __name__ == "__main__":
    main()
