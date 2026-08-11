"""Small, proxy-free client for Label Studio project snapshots and updates."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LabelStudioProjectSnapshot:
    project: dict[str, Any]
    tasks: list[dict[str, Any]]


class LabelStudioProjectApi:
    """Use a server-side API token without exposing it to the browser."""

    def __init__(self, *, base_url: str, token: str, timeout_seconds: int = 120) -> None:
        if not base_url.strip():
            raise ValueError("Label Studio URL is required")
        if not token.strip():
            raise ValueError("Label Studio API token is required")
        if timeout_seconds <= 0:
            raise ValueError("Label Studio timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = {"Authorization": f"Token {self.token}"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Label Studio API {method} {path} failed: {error.code} {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Label Studio API {method} {path} failed: {error.reason}"
            ) from error
        return json.loads(raw) if raw else {}

    def snapshot(self, project_id: int) -> LabelStudioProjectSnapshot:
        if project_id <= 0:
            raise ValueError("Label Studio project ID must be positive")
        project = self.request(method="GET", path=f"/api/projects/{project_id}")
        tasks = self.request(
            method="GET",
            path=f"/api/projects/{project_id}/export?exportType=JSON",
        )
        if not isinstance(project, dict):
            raise ValueError("Label Studio project response is not an object")
        if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
            raise ValueError("Label Studio export response is not an array of tasks")
        project_response_id = int(project.get("id", 0))
        if project_response_id != project_id:
            raise ValueError(
                f"Label Studio returned project {project_response_id}, expected {project_id}"
            )
        for position, task in enumerate(tasks, start=1):
            if int(task.get("project", 0)) != project_id:
                raise ValueError(
                    f"Label Studio export task at position {position} belongs to another project"
                )
        return LabelStudioProjectSnapshot(project=project, tasks=tasks)

    def patch(self, resource: str, object_id: int, payload: dict[str, Any]) -> Any:
        if resource not in {"projects", "tasks", "annotations", "drafts"}:
            raise ValueError(f"unsupported Label Studio resource: {resource}")
        if object_id <= 0:
            raise ValueError("Label Studio object ID must be positive")
        return self.request(
            method="PATCH",
            path=f"/api/{resource}/{object_id}",
            payload=payload,
        )


def snapshot_to_sync_state(snapshot: LabelStudioProjectSnapshot) -> dict[str, Any]:
    """Convert a native API export into the state expected by K-fold sync."""
    label_config = snapshot.project.get("label_config")
    if not isinstance(label_config, str) or not label_config.strip():
        raise ValueError("Label Studio project has no label configuration")
    tasks: dict[int, dict[str, Any]] = {}
    annotations: dict[int, list[tuple[int, list[dict[str, Any]]]]] = {}
    drafts: dict[int, list[tuple[int, list[dict[str, Any]]]]] = {}
    for task in snapshot.tasks:
        task_id = int(task.get("id", 0))
        if task_id <= 0 or task_id in tasks:
            raise ValueError(f"invalid or duplicate Label Studio task ID: {task_id}")
        data = task.get("data")
        if not isinstance(data, dict):
            raise ValueError(f"Label Studio task {task_id} data is not an object")
        tasks[task_id] = data
        for annotation in task.get("annotations", []):
            if not isinstance(annotation, dict) or annotation.get("was_cancelled", False):
                continue
            results = annotation.get("result", [])
            if not isinstance(results, list):
                raise ValueError(f"Label Studio annotation in task {task_id} has invalid results")
            annotations.setdefault(task_id, []).append((int(annotation["id"]), results))
        for draft in task.get("drafts", []):
            if not isinstance(draft, dict):
                continue
            results = draft.get("result", [])
            if not isinstance(results, list):
                raise ValueError(f"Label Studio draft in task {task_id} has invalid results")
            drafts.setdefault(task_id, []).append((int(draft["id"]), results))
    return {
        "label_config": label_config,
        "tasks": tasks,
        "annotations": annotations,
        "drafts": drafts,
    }
