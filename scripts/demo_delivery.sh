#!/usr/bin/env bash
# 特征打包与交付的端到端演示。
# 前提：平台已在 $BASE_URL 运行（docker compose up 或本地 uvicorn）。
# 用法：bash scripts/demo_delivery.sh
set -euo pipefail

BASE="${BASE_URL:-http://localhost:8000}"
WORK="$(mktemp -d /tmp/delivery-demo.XXXXXX)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
echo "== 工作目录：$WORK；服务：$BASE"

jqget() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)"; }

echo; echo "== 0. 健康检查"
curl -sf "$BASE/api/health" && echo

echo; echo "== 1. 上传示例病例与亲属关系，设置精确配额 (2,2,3,3)"
curl -sf -F "file=@$REPO_DIR/samples/cases_sample.csv" "$BASE/api/upload/cases" && echo
curl -sf -F "file=@$REPO_DIR/samples/relatives_sample.csv" "$BASE/api/upload/relatives" && echo
curl -sf -X PUT -H 'Content-Type: application/json' \
  -d '{"g1_label0":2,"g1_label1":2,"g2_label0":3,"g2_label1":3}' "$BASE/api/quotas" && echo

echo; echo "== 2. 求解并发布清单"
TOKEN=$(curl -sf -X POST "$BASE/api/solve" | jqget "['preview_token']")
MID=$(curl -sf -X POST -H 'Content-Type: application/json' \
  -d "{\"preview_token\":\"$TOKEN\"}" "$BASE/api/publish" | jqget "['manifest_id']")
echo "清单 ID：$MID"

echo; echo "== 3. 生成特征 ZIP 并打包（两包共用同一交付批次号）"
python3 "$SCRIPT_DIR/make_features_zip.py" "$MID" "$WORK/features.zip" --base "$BASE"
B1=$(curl -sf -F "manifest_id=$MID" -F "file=@$WORK/features.zip" "$BASE/api/deliveries" \
     | jqget "['batch']['batch_id']")
echo "交付批次号 B1：$B1"

echo; echo "== 4. 下载训练包与评估包并核对内容"
curl -sf -o "$WORK/train.zip" "$BASE/api/deliveries/$B1/download/train"
curl -sf -o "$WORK/eval.zip"  "$BASE/api/deliveries/$B1/download/eval"
python3 - "$WORK" <<'PY'
import json, sys, zipfile
work = sys.argv[1]
metas, sets = {}, {}
for pkg in ("train", "eval"):
    with zipfile.ZipFile(f"{work}/{pkg}.zip") as zf:
        metas[pkg] = json.loads(zf.read("delivery_batch.json"))
        sets[pkg] = sorted(n for n in zf.namelist()
                           if n.endswith(".json") and n != "delivery_batch.json")
        print(f"  {pkg}.zip: {metas[pkg]['case_count']} 例 -> {sets[pkg]}")
        print(f"    cases.csv 前几行: "
              f"{zf.read('cases.csv').decode('utf-8-sig').splitlines()[:3]}")
assert metas["train"]["batch_id"] == metas["eval"]["batch_id"], "两包批次号必须相同"
print(f"  两包共用交付批次号：{metas['train']['batch_id']}")
PY

echo; echo "== 5. 重新打包失败（故意缺件）-> 整包拒收，上一对有效包仍可下载"
python3 "$SCRIPT_DIR/make_features_zip.py" "$MID" "$WORK/broken.zip" --base "$BASE" --omit C003
HTTP=$(curl -s -o "$WORK/reject.json" -w "%{http_code}" \
  -F "manifest_id=$MID" -F "file=@$WORK/broken.zip" "$BASE/api/deliveries")
echo "拒收响应 HTTP $HTTP：$(cat "$WORK/reject.json")"
[ "$HTTP" = "400" ]
curl -sf -o /dev/null "$BASE/api/deliveries/$B1/download/train" \
  && echo "B1 训练包仍可下载 ✓"

echo; echo "== 6. 重新打包成功 -> 旧批次链接提示失效"
B2=$(curl -sf -F "manifest_id=$MID" -F "file=@$WORK/features.zip" "$BASE/api/deliveries" \
     | jqget "['batch']['batch_id']")
echo "新交付批次号 B2：$B2"
HTTP=$(curl -s -o "$WORK/stale.json" -w "%{http_code}" "$BASE/api/deliveries/$B1/download/train")
echo "旧批次 B1 下载：HTTP $HTTP — $(jqget "['detail']" < "$WORK/stale.json")"
[ "$HTTP" = "409" ]
curl -sf -o /dev/null "$BASE/api/deliveries/$B2/download/eval" && echo "B2 评估包可下载 ✓"

echo; echo "== 7. 更正切分输入 -> 清单失效 -> 两包一律不得发布或下载"
curl -sf -X PUT -H 'Content-Type: application/json' \
  -d '{"locks":[{"case_id":"C001","group":2}]}' "$BASE/api/locks" > /dev/null
HTTP=$(curl -s -o "$WORK/inv1.json" -w "%{http_code}" "$BASE/api/deliveries/$B2/download/train")
echo "B2 训练包下载：HTTP $HTTP — $(jqget "['detail']" < "$WORK/inv1.json")"
[ "$HTTP" = "409" ]
HTTP=$(curl -s -o "$WORK/inv2.json" -w "%{http_code}" \
  -F "manifest_id=$MID" -F "file=@$WORK/features.zip" "$BASE/api/deliveries")
echo "失效清单再打包：HTTP $HTTP — $(jqget "['detail']" < "$WORK/inv2.json")"
[ "$HTTP" = "409" ]

echo; echo "== 演示完成。工作目录 $WORK 保留了下载的两个 ZIP 供检查。"
