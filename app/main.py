"""FastAPI 入口：路由、预览令牌缓存、失效管控、特征交付打包。"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import Database, StaleVersionError
from .delivery import (
    MAX_UPLOAD_BYTES,
    build_packages,
    read_features_zip,
    unsafe_case_ids,
)
from .models import (
    ActiveInput,
    ValidationError,
    parse_cases_csv,
    parse_relatives_csv,
    validate_locks,
    validate_quotas,
)
from .splitter import SplitterError, solve_split

DB_PATH = Path(os.environ.get("SPLIT_DB_PATH",
                              Path(__file__).resolve().parent.parent / "data" / "app.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 交付包（train.zip / eval.zip）与 SQLite 同卷持久化，重启后仍按同一套规则供数
DELIVERIES_DIR = Path(os.environ.get("SPLIT_DELIVERIES_DIR", DB_PATH.parent / "deliveries"))
DELIVERIES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="病例切分平台", version="1.1.0")
db = Database(str(DB_PATH))


def _reconcile_delivery_storage() -> None:
    """启动对账：清理打包中途崩溃留下的临时目录，以及没有任何数据库记录的
    孤儿批次目录（文件先于事务写入、事务失败即删；崩溃才会留下孤儿）。
    有数据库记录的批次目录一律保留，重启后按原规则继续供数或拒绝。"""
    known = db.delivery_batch_ids()
    for child in DELIVERIES_DIR.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("tmp-") or child.name not in known:
            shutil.rmtree(child, ignore_errors=True)


_reconcile_delivery_storage()


# --------------------------------------------------------------------------- #
# 预览缓存：仅存内存。重启即清空——重启后不存在任何“旧预览”可被发布。
# 每个预览绑定其求解时的数据版本；版本一旦变动，令牌立即作废。
# --------------------------------------------------------------------------- #
_PREVIEW_MAX = 64
_previews: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_preview_lock = threading.Lock()


def _add_preview(version: int, input_json: dict, result: dict) -> str:
    token = uuid.uuid4().hex
    with _preview_lock:
        _previews[token] = {
            "version": version,
            "input_json": input_json,
            "result": result,
            "created_at": time.time(),
        }
        _previews.move_to_end(token)
        while len(_previews) > _PREVIEW_MAX:
            _previews.popitem(last=False)
    return token


def _get_preview(token: str) -> Dict[str, Any]:
    with _preview_lock:
        preview = _previews.get(token)
        if preview is None:
            raise StaleVersionError("该预览不存在或已因重启/过期被清除，不能发布，请重新求解。")
        current_version = db.get_version()
        if preview["version"] != current_version:
            raise StaleVersionError(
                f"预览依据的输入版本 v{preview['version']} 已变动为"
                f" v{current_version}，该预览已永久失效，不能发布，请重新求解。"
            )
        return preview


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class QuotasPayload(BaseModel):
    g1_label0: int = Field(ge=0)
    g1_label1: int = Field(ge=0)
    g2_label0: int = Field(ge=0)
    g2_label1: int = Field(ge=0)


class LocksPayload(BaseModel):
    locks: list  # [{"case_id": str, "group": 1|2}, ...]


class PublishPayload(BaseModel):
    preview_token: str


# --------------------------------------------------------------------------- #
# 异常处理
# --------------------------------------------------------------------------- #
@app.exception_handler(ValidationError)
async def _validation_handler(_request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"detail": "校验失败", "errors": exc.errors})


@app.exception_handler(SplitterError)
async def _splitter_handler(_request, exc: SplitterError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(StaleVersionError)
async def _stale_handler(_request, exc: StaleVersionError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# --------------------------------------------------------------------------- #
# 状态与上传
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "version": db.get_version()}


@app.get("/api/state")
def get_state():
    version, active = db.get_state()
    label_totals = {0: 0, 1: 0}
    for c in active.cases:
        label_totals[c.label] += 1
    return {
        "dataset_version": version,
        "label_totals": label_totals,
        "patient_count": len({c.patient_id for c in active.cases}),
        "input": active.to_json(),
        "manifests": db.list_manifests(),
        "deliveries": [_enrich_delivery(b) for b in db.list_deliveries()],
    }


async def _read_upload(file: UploadFile) -> bytes:
    raw = await file.read()
    if not raw:
        raise ValidationError([f"上传文件 {file.filename!r} 为空。"])
    if len(raw) > 2 * 1024 * 1024:
        raise ValidationError(["上传文件超过 2MB 限制。"])
    return raw


@app.post("/api/upload/cases")
async def upload_cases(file: UploadFile = File(...)):
    raw = await _read_upload(file)
    # 解析/校验全部完成且无异常后才写库；失败不触碰源数据
    cases = parse_cases_csv(raw)
    new_version = db.replace_cases(cases)
    return {"dataset_version": new_version, "case_count": len(cases),
            "notice": "病例已替换；亲属关系与指向已删除病例的锁定已清空，需要时请重新设置。"}


@app.post("/api/upload/relatives")
async def upload_relatives(file: UploadFile = File(...)):
    raw = await _read_upload(file)
    _, active = db.get_state()
    if not active.cases:
        raise ValidationError(["请先上传病例 CSV，再上传亲属关系。"])
    relatives = parse_relatives_csv(raw, [c.patient_id for c in active.cases])
    new_version = db.replace_relatives(relatives)
    return {"dataset_version": new_version, "pair_count": len(relatives)}


@app.put("/api/quotas")
def update_quotas(payload: QuotasPayload):
    _, active = db.get_state()
    totals = {0: 0, 1: 0}
    for c in active.cases:
        totals[c.label] += 1
    quotas = validate_quotas(payload.model_dump(), totals)
    new_version = db.update_quotas(quotas)
    return {"dataset_version": new_version, "quotas": payload.model_dump()}


@app.put("/api/locks")
def update_locks(payload: LocksPayload):
    _, active = db.get_state()
    locks = validate_locks(payload.locks, [c.case_id for c in active.cases])
    new_version = db.update_locks(locks)
    return {"dataset_version": new_version, "lock_count": len(locks)}


# --------------------------------------------------------------------------- #
# 求解（预览）与发布
# --------------------------------------------------------------------------- #
@app.post("/api/solve")
def solve():
    # 先取输入快照与版本；z3 计算在数据库锁之外进行，计算期间允许其他请求改输入
    version, active = db.get_state()
    if not active.cases:
        raise ValidationError(["还没有病例数据，请先上传病例 CSV。"])
    input_snapshot = active.to_json()
    result = solve_split(
        active.cases, active.relatives, active.quotas, active.locks
    )
    # 计算结束后复核版本：若期间输入被改动，本次结果对应旧版本，拒绝生成预览
    current_version = db.get_version()
    if current_version != version:
        raise HTTPException(
            status_code=409,
            detail=(
                f"计算期间输入已从 v{version} 变动到 v{current_version}，"
                "本次结果已作废且不会生成预览，请重新求解。"
            ),
        )
    token = None
    if result["status"] == "sat":
        token = _add_preview(version, input_snapshot, result)
    return {
        "dataset_version": version,
        "preview_token": token,
        "result": result,
        "input_snapshot": input_snapshot,
    }


@app.post("/api/publish")
def publish(payload: PublishPayload):
    preview = _get_preview(payload.preview_token)  # 过期/失效 -> 409
    result = preview["result"]
    if result.get("status") != "sat":
        raise HTTPException(status_code=400, detail="无解的预览不能发布。")
    manifest = db.publish_manifest(
        expected_version=preview["version"],
        input_data=preview["input_json"],
        result=result,
    )
    return manifest


@app.get("/api/manifests")
def manifests():
    return {"dataset_version": db.get_version(), "manifests": db.list_manifests()}


@app.get("/api/manifests/{manifest_id}")
def manifest_detail(manifest_id: str):
    manifest = db.get_manifest(manifest_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="清单不存在。")
    return manifest


@app.get("/api/manifests/{manifest_id}/download")
def manifest_download(manifest_id: str):
    manifest = db.get_manifest(manifest_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="清单不存在。")
    if manifest["expired"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "该清单依据的输入已变动，清单已永久失效，禁止下载。"
                "可在“已发布清单”中查看其历史记录与失效原因。"
            ),
        )
    result = manifest["result"]
    assignments = result["assignments"]
    locks = manifest["input"].get("locks", {})
    lines = ["病例号,患者号,标签,分组编号,分组名称,是否锁定"]
    for c in manifest["input"]["cases"]:
        cid = c["case_id"]
        grp = assignments[cid]
        locked = "是" if cid in locks else "否"
        lines.append(
            f"{cid},{c['patient_id']},{c['label']},{grp},"
            f"{'训练集' if grp == 1 else '评估集'},{locked}"
        )
    csv_text = "﻿" + "\n".join(lines) + "\n"
    return PlainTextResponse(
        csv_text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="split_manifest_{manifest_id[:8]}_v{manifest["dataset_version"]}.csv"'
            )
        },
    )


@app.get("/api/version-log")
def version_log():
    return {"current_version": db.get_version(), "log": db.version_log()}


# --------------------------------------------------------------------------- #
# 特征打包与交付：选有效清单 + 上传特征 ZIP -> 训练/评估两个 ZIP（同一批次号）
# --------------------------------------------------------------------------- #
def _enrich_delivery(batch: Dict[str, Any]) -> Dict[str, Any]:
    """补充文件系统状态，计算批次是否可下载及其原因。"""
    batch_dir = DELIVERIES_DIR / batch["batch_id"]
    files_exist = (batch_dir / "train.zip").is_file() and (batch_dir / "eval.zip").is_file()
    if batch["superseded"]:
        status, downloadable = "已被新批次替换", False
    elif batch["manifest_expired"]:
        status, downloadable = "清单已失效", False
    elif not files_exist:
        status, downloadable = "交付文件缺失", False
    else:
        status, downloadable = "有效", True
    return {
        **batch,
        "files_exist": files_exist,
        "status": status,
        "downloadable": downloadable,
    }


async def _read_zip_upload(file: UploadFile) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise ValidationError(
                [f"上传文件超过 {MAX_UPLOAD_BYTES // (1 << 20)} MiB 限制。"])
        chunks.append(chunk)
    if total == 0:
        raise ValidationError(["上传文件为空。"])
    return b"".join(chunks)


@app.post("/api/deliveries")
async def create_delivery(manifest_id: str = Form(...), file: UploadFile = File(...)):
    # 1) 清单必须存在且当前有效；失效清单一律不得发布交付包
    manifest = db.get_manifest(manifest_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="清单不存在。")
    if manifest["expired"]:
        raise HTTPException(
            status_code=409,
            detail="该清单已失效（切分输入已变动），两包一律不得发布。"
                   "请重新求解并发布新清单后再打包。",
        )
    case_ids = [c["case_id"] for c in manifest["input"]["cases"]]
    bad = unsafe_case_ids(case_ids)
    if bad:
        raise ValidationError([
            "以下病例号无法作为 ZIP 根目录安全文件名（含非法字符或与元数据文件"
            "冲突），不能打包：" + "、".join(bad)
        ])
    # 2) 整包校验：缺件 / 未知病例 / 同名文件 / 超量 / 非 JSON —— 任一即拒收
    raw = await _read_zip_upload(file)
    features = read_features_zip(raw, case_ids)
    # 3) 构建两包并落盘。全程不持有数据库锁：打包期间其他窗口仍可更正切分输入
    batch_id = uuid.uuid4().hex
    tmp_dir = DELIVERIES_DIR / f"tmp-{batch_id}"
    final_dir = DELIVERIES_DIR / batch_id
    try:
        pkgs = build_packages(batch_id, manifest, features, tmp_dir)
        os.replace(tmp_dir, final_dir)  # 两包一起原子就位
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(final_dir, ignore_errors=True)
        raise
    # 4) 两包都已保存成功，才在事务内提交批次记录并开放下载；
    #    提交时复核清单有效性（打包期间输入可能已被更正）
    try:
        replaced = db.commit_delivery_batch(
            batch_id=batch_id,
            manifest_id=manifest_id,
            expected_version=manifest["dataset_version"],
            train=pkgs["train"],
            eval=pkgs["eval"],
        )
    except Exception:
        shutil.rmtree(final_dir, ignore_errors=True)  # 未提交成功：两包一律不保留
        raise
    # 5) 替换成功后再清理旧批次文件；旧链接由数据库中的 superseded 标记提示失效
    for old_id in replaced:
        shutil.rmtree(DELIVERIES_DIR / old_id, ignore_errors=True)
    batch = _enrich_delivery(db.get_delivery(batch_id))
    return {"notice": "训练包与评估包均已保存成功，下载已开放。", "batch": batch}


@app.get("/api/deliveries")
def deliveries():
    return {"deliveries": [_enrich_delivery(b) for b in db.list_deliveries()]}


@app.get("/api/deliveries/{batch_id}")
def delivery_detail(batch_id: str):
    batch = db.get_delivery(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="交付批次不存在。")
    return _enrich_delivery(batch)


@app.get("/api/deliveries/{batch_id}/download/{package}")
def delivery_download(batch_id: str, package: str):
    if package not in ("train", "eval"):
        raise HTTPException(status_code=404, detail="未知的包类型（应为 train 或 eval）。")
    batch = db.get_delivery(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="交付批次不存在。")
    if batch["superseded"]:
        raise HTTPException(
            status_code=409,
            detail="该交付批次已被同一清单的新批次替换，此下载链接已失效。"
                   "请使用最新批次的下载链接。",
        )
    if batch["manifest_expired"]:
        raise HTTPException(
            status_code=409,
            detail="该批次所属清单已失效（切分输入已变动），两包一律不得下载。",
        )
    batch_dir = DELIVERIES_DIR / batch_id
    train_zip = batch_dir / "train.zip"
    eval_zip = batch_dir / "eval.zip"
    if not (train_zip.is_file() and eval_zip.is_file()):
        raise HTTPException(
            status_code=409,
            detail="交付文件缺失（两包未同时保存成功），禁止下载。",
        )
    path = train_zip if package == "train" else eval_zip
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"{package}_features_batch_{batch_id[:8]}.zip",
    )


# 静态前端（最后挂载，避免覆盖 /api 路由）
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
