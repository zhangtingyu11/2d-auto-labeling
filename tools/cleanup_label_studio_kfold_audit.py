"""Remove K-fold audit UI and results from a Label Studio project via its API."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


AUDIT_DATA_KEYS = {"review_tag", "review_tag_label", "review_summary"}
AUDIT_RESULT_CONTROLS = {"prediction_reference", "review_issue"}
AUDIT_CONFIG_NAMES = {
    "review_tag",
    "review_summary",
    "review_issue",
    "prediction_reference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-task-data",
        action="store_true",
        help="Leave hidden task audit metadata in place and remove only visible audit UI/results",
    )
    parser.add_argument(
        "--direct-json-cleanup",
        action="store_true",
        help="Clean task/result JSON in one SQLite transaction, then update config via API",
    )
    parser.add_argument(
        "--reuse-backup",
        action="store_true",
        help="Reuse an existing integrity-checked backup after a failed API attempt",
    )
    return parser.parse_args()


def clean_task_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if key not in AUDIT_DATA_KEYS and not key.startswith("kfold_")
    }


def clean_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in results if row.get("from_name") not in AUDIT_RESULT_CONTROLS]


def clean_label_config(label_config: str) -> str:
    root = ET.fromstring(label_config)
    for parent in root.iter():
        for child in list(parent):
            name = child.attrib.get("name")
            value = child.attrib.get("value", "")
            if name in AUDIT_CONFIG_NAMES or "$review_tag_label" in value:
                parent.remove(child)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _backup_database(source: Path, destination: Path, *, reuse_existing: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def _load_state(
    database: Path, project_id: int
) -> tuple[str, str, list[dict], list[dict], list[dict]]:
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        project = connection.execute(
            "SELECT label_config, created_by_id FROM project WHERE id = ?", (project_id,)
        ).fetchone()
        if project is None:
            raise ValueError(f"project {project_id} does not exist")
        label_config, owner_id = project
        token_row = connection.execute(
            "SELECT key FROM authtoken_token WHERE user_id = ? ORDER BY created DESC LIMIT 1",
            (owner_id,),
        ).fetchone()
        if token_row is None:
            raise RuntimeError(f"project owner {owner_id} has no API token")
        tasks = [
            {"id": int(task_id), "data": json.loads(raw_data)}
            for task_id, raw_data in connection.execute(
                "SELECT id, data FROM task WHERE project_id = ? ORDER BY id", (project_id,)
            )
        ]
        annotations = [
            {"id": int(annotation_id), "result": json.loads(raw_results or "[]")}
            for annotation_id, raw_results in connection.execute(
                "SELECT id, result FROM task_completion WHERE project_id = ? ORDER BY id",
                (project_id,),
            )
        ]
        drafts = [
            {"id": int(draft_id), "result": json.loads(raw_results or "[]")}
            for draft_id, raw_results in connection.execute(
                """
                SELECT draft.id, draft.result
                FROM tasks_annotationdraft AS draft
                JOIN task ON task.id = draft.task_id
                WHERE task.project_id = ?
                ORDER BY draft.id
                """,
                (project_id,),
            )
        ]
        return token_row[0], label_config, tasks, annotations, drafts
    finally:
        connection.close()


def _request(
    *,
    url: str,
    token: str,
    method: str,
    payload: dict[str, Any],
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


def _get_json(*, url: str, token: str) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with opener.open(request, timeout=30) as response:
        return json.loads(response.read())


def _run_updates(
    updates: list[tuple[int, Any]],
    *,
    resource: str,
    payload_key: str,
    base_url: str,
    token: str,
    workers: int,
    progress_every: int,
) -> None:
    def update(item: tuple[int, Any]) -> None:
        object_id, value = item
        _request(
            url=f"{base_url}/api/{resource}/{object_id}",
            token=token,
            method="PATCH",
            payload={payload_key: value},
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(update, item) for item in updates]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            if index % progress_every == 0 or index == len(updates):
                print(f"{resource}={index}/{len(updates)}", flush=True)


def _update_json_in_transaction(
    database: Path,
    tasks: list[tuple[int, dict[str, Any]]],
    annotations: list[tuple[int, list[dict[str, Any]]]],
) -> None:
    connection = sqlite3.connect(database, timeout=30)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            "UPDATE task SET data = ? WHERE id = ?",
            [
                (json.dumps(data, ensure_ascii=False, separators=(",", ":")), task_id)
                for task_id, data in tasks
            ],
        )
        connection.executemany(
            "UPDATE task_completion SET result = ? WHERE id = ?",
            [
                (
                    json.dumps(results, ensure_ascii=False, separators=(",", ":")),
                    annotation_id,
                )
                for annotation_id, results in annotations
            ],
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    token, label_config, tasks, annotations, drafts = _load_state(
        args.database, args.project_id
    )
    clean_config = clean_label_config(label_config)
    changed_tasks = [
        (task["id"], clean_task_data(task["data"]))
        for task in tasks
        if clean_task_data(task["data"]) != task["data"]
    ]
    if args.skip_task_data:
        changed_tasks = []
    changed_annotations = [
        (annotation["id"], clean_results(annotation["result"]))
        for annotation in annotations
        if clean_results(annotation["result"]) != annotation["result"]
    ]
    changed_drafts = [
        (draft["id"], clean_results(draft["result"]))
        for draft in drafts
        if clean_results(draft["result"]) != draft["result"]
    ]
    _backup_database(args.database, args.backup, reuse_existing=args.reuse_backup)
    print(f"backup={args.backup}", flush=True)

    base_url = args.url.rstrip("/")
    # Label Studio refuses to remove controls while stored results still use them.
    if args.direct_json_cleanup:
        _update_json_in_transaction(args.database, changed_tasks, changed_annotations)
        print(f"annotations={len(changed_annotations)}/{len(changed_annotations)}", flush=True)
        print(f"tasks={len(changed_tasks)}/{len(changed_tasks)}", flush=True)
    else:
        _run_updates(
            changed_annotations,
            resource="annotations",
            payload_key="result",
            base_url=base_url,
            token=token,
            workers=args.workers,
            progress_every=50,
        )
        _run_updates(
            changed_drafts,
            resource="drafts",
            payload_key="result",
            base_url=base_url,
            token=token,
            workers=1,
            progress_every=10,
        )
        _run_updates(
            changed_tasks,
            resource="tasks",
            payload_key="data",
            base_url=base_url,
            token=token,
            workers=args.workers,
            progress_every=100,
        )

    if clean_config != label_config:
        _request(
            url=f"{base_url}/api/projects/{args.project_id}/summary/reset/",
            token=token,
            method="POST",
            payload={},
        )
        summary_url = f"{base_url}/api/projects/{args.project_id}/summary/"
        for _ in range(30):
            summary = _get_json(url=summary_url, token=token)
            created = json.dumps(
                {
                    "annotations": summary.get("created_annotations", {}),
                    "labels": summary.get("created_labels", {}),
                    "drafts": summary.get("created_labels_drafts", {}),
                }
            )
            if not any(control in created for control in AUDIT_RESULT_CONTROLS):
                break
            time.sleep(2)
        else:
            raise RuntimeError("Label Studio project summary reset did not finish in 60 seconds")
        _request(
            url=f"{base_url}/api/projects/{args.project_id}",
            token=token,
            method="PATCH",
            payload={"label_config": clean_config},
        )
        print("project_config=cleaned", flush=True)

    print(
        json.dumps(
            {
                "project_id": args.project_id,
                "cleaned_tasks": len(changed_tasks),
                "cleaned_annotations": len(changed_annotations),
                "cleaned_drafts": len(changed_drafts),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise
