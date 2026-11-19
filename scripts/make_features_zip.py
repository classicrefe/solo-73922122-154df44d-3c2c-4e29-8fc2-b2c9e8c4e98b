#!/usr/bin/env python3
"""从运行中的平台拉取指定清单的病例列表，生成演示用特征 ZIP。

用法：
    python3 scripts/make_features_zip.py <manifest_id> <输出.zip> [--base URL] [--omit 病例号]

清单中每个病例在 ZIP 根目录生成一个 ``病例号.json``，内容为确定性的演示特征
（由病例号哈希派生，便于重复演示时比对）。--omit 可故意漏掉某例，用于演示
“缺件整包拒收”。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_id")
    ap.add_argument("out_zip")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--omit", default=None, help="故意漏掉该病例（演示缺件拒收）")
    args = ap.parse_args()

    with urllib.request.urlopen(f"{args.base}/api/manifests/{args.manifest_id}") as r:
        manifest = json.load(r)
    if manifest.get("expired"):
        print("清单已失效，不能为其生成特征包。", file=sys.stderr)
        return 1
    cases = manifest["input"]["cases"]

    with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in cases:
            cid = c["case_id"]
            if cid == args.omit:
                continue
            seed = hashlib.sha256(cid.encode()).digest()
            features = [round(b / 255, 4) for b in seed[:16]]
            payload = {
                "case_id": cid,
                "patient_id": c["patient_id"],
                "modality": "CT",
                "features": features,
            }
            zf.writestr(f"{cid}.json",
                        json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"已生成 {args.out_zip}：{len(cases)} 例"
          + (f"（故意漏掉 {args.omit}）" if args.omit else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
