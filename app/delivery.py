"""特征 ZIP 校验与训练/评估交付包构建。

规则（任何一条不满足都整包拒收，不产生任何交付物）：
- 上传 ZIP 的根目录下必须是每例一个 ``病例号.json``，文件集合与所选清单的
  病例集合完全一致：缺件、未知病例（或任何非 ``病例号.json`` 的文件）、
  同名文件都整包拒收；
- 解压总量（所有条目实际解压出的字节数）不得超过 8 MiB；
- 每个病例文件必须是非空的合法 JSON；
- 按清单的切分结果把病例文件分别写入 train.zip / eval.zip，两包内各附
  cases.csv（病例列表）与 delivery_batch.json（含同一个交付批次号）。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from .models import ValidationError

MAX_EXTRACTED_BYTES = 8 * 1024 * 1024   # 解压总量上限 8 MiB
MAX_UPLOAD_BYTES = 16 * 1024 * 1024     # 上传压缩包本身的防御性上限
MAX_ZIP_ENTRIES = 1000                  # 条目数防御性上限（合法包不超过病例数）

_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
# 交付包内的元数据文件名（词干），病例号不得与之冲突
RESERVED_STEMS = {"cases", "delivery_batch"}

PACKAGE_NAMES = {"train": "训练集", "eval": "评估集"}


def unsafe_case_ids(case_ids: Sequence[str]) -> List[str]:
    """病例号将直接用作 ZIP 根目录文件名，必须文件名安全且不与元数据冲突。"""
    bad = []
    for cid in case_ids:
        if (not _SAFE_CASE_ID.match(cid)) or cid in (".", "..") or cid in RESERVED_STEMS:
            bad.append(cid)
    return sorted(bad)


def read_features_zip(raw: bytes, required_case_ids: Sequence[str]) -> Dict[str, bytes]:
    """校验并读取特征 ZIP，返回 {病例号: 文件原始字节}（字节原样保留）。

    所有结构问题汇总为 ValidationError 一次性报出；任何错误都整包拒收。
    """
    required_names = {f"{cid}.json" for cid in required_case_ids}
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValidationError([f"上传文件不是有效的 ZIP：{exc}"])
    with zf:
        entries = [i for i in zf.infolist() if not i.is_dir()]
        if len(entries) > MAX_ZIP_ENTRIES:
            raise ValidationError(
                [f"ZIP 条目数 {len(entries)} 超过上限 {MAX_ZIP_ENTRIES}，整包拒收。"])

        errors: List[str] = []
        encrypted = sorted({i.filename for i in entries if i.flag_bits & 0x1})
        if encrypted:
            errors.append("存在加密条目，不予接受：" + "、".join(encrypted[:5]))

        names = [i.filename for i in entries]
        seen, dupes = set(), set()
        for n in names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            errors.append("存在同名文件：" + "、".join(sorted(dupes)[:10]))

        name_set = set(names)
        missing = sorted(required_names - name_set)
        unknown = sorted(name_set - required_names)
        if missing:
            shown = "、".join(missing[:10]) + (" …" if len(missing) > 10 else "")
            errors.append(f"缺件 {len(missing)} 例：{shown}")
        if unknown:
            shown = "、".join(unknown[:10]) + (" …" if len(unknown) > 10 else "")
            errors.append(
                f"存在清单外文件 {len(unknown)} 个（未知病例、非根目录文件或非“病例号.json”）：{shown}")

        declared = sum(i.file_size for i in entries)
        if declared > MAX_EXTRACTED_BYTES:
            errors.append(
                f"解压总量 {declared} 字节超过 8 MiB 上限（{MAX_EXTRACTED_BYTES} 字节）。")
        if errors:
            raise ValidationError(errors)

        # 逐条读取并按实际字节复核总量（防御目录中虚报大小的压缩包）；
        # zipfile 读取时自动做 CRC 校验，损坏条目在此被捕获。
        features: Dict[str, bytes] = {}
        total = 0
        try:
            for info in entries:
                with zf.open(info) as fh:
                    buf = []
                    while True:
                        chunk = fh.read(1 << 16)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_EXTRACTED_BYTES:
                            raise ValidationError(["解压总量超过 8 MiB 上限，整包拒收。"])
                        buf.append(chunk)
                features[info.filename] = b"".join(buf)
        except zipfile.BadZipFile as exc:
            raise ValidationError([f"ZIP 内容损坏（CRC 校验失败）：{exc}"])

    json_errors = []
    for name in sorted(features):
        data = features[name]
        if not data:
            json_errors.append(f"{name} 内容为空")
            continue
        try:
            json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            json_errors.append(f"{name} 不是合法 JSON：{exc}")
    if json_errors:
        raise ValidationError(json_errors)
    return {name[: -len(".json")]: data for name, data in features.items()}


def build_packages(batch_id: str, manifest: dict, features: Dict[str, bytes],
                   work_dir: Path) -> Dict[str, dict]:
    """在 work_dir（必须尚不存在）中构建 train.zip 与 eval.zip。

    每个包包含：该组全部 ``病例号.json``（字节与上传一致）、cases.csv
    （病例列表）、delivery_batch.json（两包共用同一个 batch_id）。
    返回 {"train": {...}, "eval": {...}}（病例列表、数量、字节数、SHA-256）。
    """
    cases = manifest["input"]["cases"]
    assignments = manifest["result"]["assignments"]
    meta_by_id = {c["case_id"]: c for c in cases}
    groups = {"train": [], "eval": []}
    for cid in sorted(features):
        grp = assignments.get(cid)
        if grp not in (1, 2):
            raise ValidationError([f"清单切分结果中缺少病例 {cid} 的分组，无法打包。"])
        groups["train" if grp == 1 else "eval"].append(cid)

    work_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: Dict[str, dict] = {}
    for pkg in ("train", "eval"):
        case_ids = groups[pkg]
        path = work_dir / f"{pkg}.zip"
        meta = {
            "batch_id": batch_id,
            "package": pkg,
            "package_name": PACKAGE_NAMES[pkg],
            "manifest_id": manifest["manifest_id"],
            "dataset_version": manifest["dataset_version"],
            "case_count": len(case_ids),
            "created_at": created_at,
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("delivery_batch.json",
                        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
            zf.writestr("cases.csv", _cases_csv([meta_by_id[c] for c in case_ids]))
            for cid in case_ids:
                zf.writestr(f"{cid}.json", features[cid])
        blob = path.read_bytes()
        out[pkg] = {
            "cases": case_ids,
            "count": len(case_ids),
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
    return out


def _cases_csv(cases: Sequence[dict]) -> str:
    buf = io.StringIO()
    buf.write("﻿")  # BOM，便于 Excel 直接打开
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["病例号", "患者号", "标签"])
    for c in cases:
        writer.writerow([c["case_id"], c["patient_id"], c["label"]])
    return buf.getvalue()
