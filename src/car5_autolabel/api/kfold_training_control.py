"""Authenticated web control plane for human-reviewed K-fold training jobs."""

# ruff: noqa: E501 -- The self-contained HTML/JavaScript page is intentionally inline.

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}")
ACTIVE_STATES = {"queued", "exporting", "preparing", "running", "syncing"}


@dataclass(frozen=True)
class TrainingControlConfig:
    assignment_template: Path
    image_root: Path
    workspace_root: Path
    artifact_root: Path
    repository_root: Path
    username: str
    password: str
    docker_image: str = "car5-rfdetr:v4.4-70px-train-user"
    project_id: int = 3
    gpu_indices: tuple[int, ...] = (3, 4)
    batch_size: int = 32
    grad_accum_steps: int = 1
    epochs: int = 80
    early_stopping_patience: int = 15
    idle_confirmations: int = 1
    poll_seconds: int = 5

    def validate(self) -> None:
        if not self.assignment_template.is_file():
            raise ValueError(f"assignment template does not exist: {self.assignment_template}")
        if not self.image_root.is_dir():
            raise ValueError(f"image root does not exist: {self.image_root}")
        if not (self.repository_root / "tools/run_rfdetr_kfold_label_audit.py").is_file():
            raise ValueError(f"repository root is invalid: {self.repository_root}")
        if not self.username or not self.password:
            raise ValueError("training control username and password are required")
        if self.batch_size <= 0 or self.grad_accum_steps <= 0 or self.epochs <= 0:
            raise ValueError("training parameters must be positive")
        if not self.gpu_indices or any(index < 0 for index in self.gpu_indices):
            raise ValueError("at least one non-negative GPU index is required")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"job metadata is not an object: {path}")
    return value


class TrainingJobManager:
    """Create versioned jobs and keep one fixed K-fold runner active at a time."""

    def __init__(self, config: TrainingControlConfig) -> None:
        self.config = config
        self.config.validate()
        self.jobs_root = config.artifact_root / "control_jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        self.config.artifact_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _metadata_path(self, job_id: str) -> Path:
        return self.jobs_root / job_id / "job.json"

    def _all_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for path in sorted(self.jobs_root.glob("*/job.json"), reverse=True):
            try:
                jobs.append(_read_json(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return jobs

    def _refresh(self, metadata: dict[str, Any]) -> dict[str, Any]:
        # The control service may run in a container while the worker and its
        # training child run in the host PID namespace.  A host PID therefore
        # cannot be probed reliably from here.  The host worker is the sole
        # owner of lifecycle state; the web process only reports what it wrote.
        return metadata

    def list_jobs(self) -> list[dict[str, Any]]:
        return [self._refresh(item) for item in self._all_jobs()]

    def get_job(self, job_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            raise KeyError(job_id)
        path = self._metadata_path(job_id)
        if not path.is_file():
            raise KeyError(job_id)
        return self._refresh(_read_json(path))

    def _active_job(self) -> dict[str, Any] | None:
        return next(
            (job for job in self.list_jobs() if job.get("state") in ACTIVE_STATES), None
        )

    def create_job(self, *, run_name: str) -> dict[str, Any]:
        if not SAFE_RUN_NAME.fullmatch(run_name):
            raise ValueError(
                "run name must contain 1-48 ASCII letters, digits, hyphens, or underscores"
            )
        with self._lock:
            active = self._active_job()
            if active is not None:
                raise RuntimeError(
                    f"job {active['job_id']} is already {active['state']}; only one job is allowed"
                )
            timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            job_id = f"{timestamp}-{run_name}-{uuid.uuid4().hex[:8]}"
            job_dir = self.jobs_root / job_id
            job_dir.mkdir(parents=False)
            workspace = self.config.workspace_root / job_id
            artifacts = self.config.artifact_root / "runs" / job_id
            metadata = {
                "job_id": job_id,
                "run_name": run_name,
                "state": "queued",
                "source_mode": "label_studio_api",
                "label_studio_project_id": self.config.project_id,
                "created_at": datetime.now().astimezone().isoformat(),
                "queued_at": datetime.now().astimezone().isoformat(),
                "workspace_directory": str(workspace),
                "artifact_directory": str(artifacts),
                "log_path": str(job_dir / "runner.log"),
                "docker_image": self.config.docker_image,
                "batch_size": self.config.batch_size,
                "grad_accum_steps": self.config.grad_accum_steps,
                "gpu_indices": list(self.config.gpu_indices),
            }
            _atomic_json(self._metadata_path(job_id), metadata)
            return metadata

    def read_log(self, job_id: str, *, maximum_bytes: int = 100_000) -> str:
        metadata = self.get_job(job_id)
        path = Path(metadata["log_path"])
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes))
            return handle.read().decode("utf-8", errors="replace")


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>K折训练 · Label Studio</title>
<style>
body{margin:0;background:#f6f7f9;color:#1f2937;font:14px system-ui,-apple-system,"Segoe UI",sans-serif}.top{height:56px;background:#111827;color:white;display:flex;align-items:center;padding:0 24px;gap:18px}.top a{color:#d1d5db;text-decoration:none}.wrap{max-width:980px;margin:28px auto;padding:0 20px}.card{background:white;border:1px solid #e5e7eb;border-radius:10px;padding:22px;margin-bottom:18px;box-shadow:0 1px 2px #0000000a}h1{font-size:20px;margin:0}.hint{color:#64748b;line-height:1.7}.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}label{display:block;font-weight:600;margin-bottom:7px}input{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cbd5e1;border-radius:6px}button{background:#7c3aed;color:white;border:0;border-radius:6px;padding:11px 18px;font-weight:700;cursor:pointer}button:disabled{opacity:.5}.status{display:flex;gap:12px;align-items:center}.pill{padding:4px 9px;border-radius:999px;background:#ede9fe;color:#5b21b6;font-weight:700}pre{background:#111827;color:#d1fae5;padding:14px;border-radius:7px;max-height:430px;overflow:auto;white-space:pre-wrap}.error{color:#b91c1c}.ok{color:#047857}@media(max-width:700px){.row{grid-template-columns:1fr}}
</style></head><body>
<div class="top"><h1>2D数据集 · K折训练</h1><a href="/projects/__PROJECT_ID__/data/">← 返回 Label Studio</a></div>
<main class="wrap"><section class="card"><h2>启动新一轮五折训练</h2>
<p class="hint">无需导出文件。点击后自动读取当前 Label Studio 项目的全部已提交标注，只保留可编辑人工框并删除锁定的“可能××”提示；五折训练成功后自动去重并更新提示。训练期间若人工标注发生变化，系统会停止回写，避免旧结果覆盖新标注。</p>
<form id="form"><div><label>训练版本名（仅用于区分本轮产物，支持英文、数字、-、_）</label><input name="run_name" value="reviewed-v2" pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,47}" required></div><p><button id="start">一键读取、训练并更新</button></p></form><div id="message"></div></section>
<section class="card"><div class="status"><h2>最近任务</h2><span id="state" class="pill">无</span></div><div id="summary" class="hint"></div><pre id="log">尚未启动训练</pre></section></main>
<script>
let current=null;const form=document.querySelector('#form'),msg=document.querySelector('#message'),log=document.querySelector('#log'),state=document.querySelector('#state'),summary=document.querySelector('#summary');
const stateNames={queued:'等待处理',exporting:'读取标注',preparing:'筛选数据',running:'五折训练中',syncing:'更新Label Studio',completed:'已完成并更新',stale:'训练完成，标注有新修改，未回写',failed:'失败',sync_failed:'训练完成，回写失败',interrupted:'训练中断'};
async function checked(url,options){const r=await fetch(url,options);if(!r.ok){let detail='请求失败';try{detail=(await r.json()).detail||detail}catch(_){}throw new Error(detail)}return r;}
async function refresh(){try{const list=await checked('/kfold-training/api/jobs').then(r=>r.json());if(!current&&list.length)current=list[0].job_id;if(!current)return;const job=await checked('/kfold-training/api/jobs/'+current).then(r=>r.json());state.textContent=stateNames[job.state]||job.state;const c=job.conversion||{};const s=job.sync||{};summary.textContent=`任务 ${job.job_id}｜人工框 ${c.human_box_count??'-'}｜已删除提示 ${c.removed_non_human_result_count??'-'}｜空图 ${c.empty_task_count??'-'}｜回写 ${s.changed_annotations??'-'} 个标注｜BS ${job.batch_size}｜GPU ${(job.gpu_indices||[]).join(',')}`;log.textContent=await checked('/kfold-training/api/jobs/'+current+'/log').then(r=>r.text());log.scrollTop=log.scrollHeight;}catch(e){msg.className='error';msg.textContent=e.message;}}
form.addEventListener('submit',async e=>{e.preventDefault();msg.textContent='任务提交中…';document.querySelector('#start').disabled=true;try{const r=await checked('/kfold-training/api/jobs',{method:'POST',body:new FormData(form)});const body=await r.json();current=body.job_id;msg.className='ok';msg.textContent='已进入队列：将自动读取标注、训练并回写';await refresh();}catch(e){msg.className='error';msg.textContent=e.message;}finally{document.querySelector('#start').disabled=false;}});refresh();setInterval(refresh,5000);
</script></body></html>"""


def create_training_control_app(config: TrainingControlConfig) -> FastAPI:
    manager = TrainingJobManager(config)
    security = HTTPBasic()
    app = FastAPI(title="car5 K-fold training control", docs_url=None, redoc_url=None)

    def authenticate(
        credentials: HTTPBasicCredentials = Depends(security),  # noqa: B008
    ) -> str:
        valid = secrets.compare_digest(
            credentials.username, config.username
        ) and secrets.compare_digest(credentials.password, config.password)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid training-control credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username

    @app.get("/kfold-training/", response_class=HTMLResponse)
    def page(_user: str = Depends(authenticate)) -> str:  # noqa: B008
        return PAGE.replace("__PROJECT_ID__", str(config.project_id))

    @app.get("/kfold-training/launcher.js")
    def launcher() -> Response:
        script = f"""
        (() => {{
          const ensure=() => {{
            const existing=document.getElementById('car5-kfold-launcher');
            if (!location.pathname.startsWith('/projects/{config.project_id}/')) {{ existing?.remove(); return; }}
            if (existing) return;
            const button=document.createElement('a'); button.id='car5-kfold-launcher';
            button.href='/kfold-training/'; button.textContent='K折训练';
            Object.assign(button.style,{{position:'fixed',right:'22px',bottom:'22px',zIndex:'2147483647',background:'#7c3aed',color:'#fff',padding:'11px 17px',borderRadius:'7px',fontWeight:'700',textDecoration:'none',boxShadow:'0 4px 14px #0004'}});
            document.body.appendChild(button);
          }};
          ensure(); setInterval(ensure,1000);
        }})();
        """
        return Response(script, media_type="application/javascript")

    @app.get("/kfold-training/api/jobs")
    def jobs(_user: str = Depends(authenticate)) -> JSONResponse:  # noqa: B008
        return JSONResponse(manager.list_jobs())

    @app.get("/kfold-training/api/jobs/{job_id}")
    def job(
        job_id: str, _user: str = Depends(authenticate)  # noqa: B008
    ) -> JSONResponse:
        try:
            return JSONResponse(manager.get_job(job_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.get("/kfold-training/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(
        job_id: str, _user: str = Depends(authenticate)  # noqa: B008
    ) -> str:
        try:
            return manager.read_log(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.post("/kfold-training/api/jobs", status_code=201)
    def create_job(
        run_name: str = Form(...),  # noqa: B008
        _user: str = Depends(authenticate),  # noqa: B008
    ) -> JSONResponse:
        try:
            metadata = manager.create_job(run_name=run_name)
            return JSONResponse(metadata, status_code=201)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app
