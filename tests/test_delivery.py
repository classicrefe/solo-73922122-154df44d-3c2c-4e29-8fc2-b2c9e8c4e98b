"""特征打包与交付测试：ZIP 校验、两包构建、批次替换、失效管控、持久化、并发。"""
import io
import json
import os
import sys
import zipfile

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES_CSV = (
    "病例号,患者号,标签\n"
    "C1,P1,1\nC2,P1,0\nC3,P2,1\nC4,P2,0\n"
    "C5,P3,1\nC6,P3,0\nC7,P4,1\nC8,P4,0\n"
)
ALL_CASES = [f"C{i}" for i in range(1, 9)]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """每个用例独立数据库与交付目录；返回 (client, main 模块, tmp_path)。"""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("SPLIT_DB_PATH", str(db_file))
    import importlib
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c, main, tmp_path
    main.db.close()


def make_zip(case_ids=ALL_CASES, omit=(), extra=None, duplicate=None,
             content_fn=None):
    """构造特征 ZIP：根目录下每例一个 病例号.json。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for cid in case_ids:
            if cid in omit:
                continue
            data = content_fn(cid) if content_fn else json.dumps(
                {"case_id": cid, "features": [0.1, 0.2]}).encode()
            zf.writestr(f"{cid}.json", data)
            if duplicate == cid:
                zf.writestr(f"{cid}.json", b"{}")  # 同名文件
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


def publish_manifest(client):
    """上传 8 病例并发布一份有效清单，返回 manifest_id。"""
    r = client.post("/api/upload/cases",
                    files={"file": ("c.csv", CASES_CSV.encode(), "text/csv")})
    assert r.status_code == 200, r.text
    r = client.put("/api/quotas", json={
        "g1_label0": 2, "g1_label1": 2, "g2_label0": 2, "g2_label1": 2})
    assert r.status_code == 200, r.text
    token = client.post("/api/solve").json()["preview_token"]
    assert token
    pub = client.post("/api/publish", json={"preview_token": token})
    assert pub.status_code == 200, pub.text
    return pub.json()["manifest_id"]


def package(client, manifest_id, zip_bytes):
    return client.post("/api/deliveries", data={"manifest_id": manifest_id},
                       files={"file": ("features.zip", zip_bytes, "application/zip")})


def package_ok(client, manifest_id, zip_bytes=None):
    r = package(client, manifest_id, zip_bytes or make_zip())
    assert r.status_code == 200, r.text
    return r.json()["batch"]


# --------------------------------------------------------------------------- #
# 打包与下载主流程
# --------------------------------------------------------------------------- #
def test_package_and_download_flow(env):
    client, main, _ = env
    mid = publish_manifest(client)
    batch = package_ok(client, mid)

    assert batch["status"] == "有效" and batch["downloadable"] is True
    assert batch["train"]["count"] + batch["eval"]["count"] == 8
    assert batch["train"]["count"] == 4 and batch["eval"]["count"] == 4

    # 两包都可下载，且内容正确
    zips = {}
    for pkg in ("train", "eval"):
        dl = client.get(f"/api/deliveries/{batch['batch_id']}/download/{pkg}")
        assert dl.status_code == 200, dl.text
        assert dl.headers["content-type"] == "application/zip"
        zips[pkg] = zipfile.ZipFile(io.BytesIO(dl.content))

    metas = {}
    case_sets = {}
    for pkg, zf in zips.items():
        names = set(zf.namelist())
        assert "delivery_batch.json" in names and "cases.csv" in names
        metas[pkg] = json.loads(zf.read("delivery_batch.json"))
        case_sets[pkg] = {n[:-5] for n in names if n.endswith(".json")
                          and n != "delivery_batch.json"}
        # 病例列表与包内病例一致
        csv_text = zf.read("cases.csv").decode("utf-8-sig")
        listed = [ln.split(",")[0] for ln in csv_text.strip().split("\n")[1:]]
        assert sorted(listed) == sorted(case_sets[pkg])
        # 特征字节与上传一致
        for cid in case_sets[pkg]:
            assert json.loads(zf.read(f"{cid}.json"))["case_id"] == cid

    # 两包共用同一个交付批次号；病例不重叠且覆盖全集
    assert metas["train"]["batch_id"] == metas["eval"]["batch_id"] == batch["batch_id"]
    assert metas["train"]["manifest_id"] == mid
    assert case_sets["train"].isdisjoint(case_sets["eval"])
    assert case_sets["train"] | case_sets["eval"] == set(ALL_CASES)


def test_package_requires_existing_valid_manifest(env):
    client, _, _ = env
    r = package(client, "0" * 32, make_zip())
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# 整包拒收：缺件 / 未知病例 / 同名文件 / 超量 / 非 JSON / 目录条目
# --------------------------------------------------------------------------- #
def test_missing_file_rejected(env):
    client, _, _ = env
    mid = publish_manifest(client)
    r = package(client, mid, make_zip(omit=("C3",)))
    assert r.status_code == 400
    assert any("缺件" in e and "C3.json" in e for e in r.json()["errors"])
    assert client.get("/api/deliveries").json()["deliveries"] == []


def test_unknown_case_rejected(env):
    client, _, _ = env
    mid = publish_manifest(client)
    r = package(client, mid, make_zip(extra={"C999.json": b"{}"}))
    assert r.status_code == 400
    assert any("清单外文件" in e and "C999.json" in e for e in r.json()["errors"])
    assert client.get("/api/deliveries").json()["deliveries"] == []


def test_duplicate_name_rejected(env):
    client, _, _ = env
    mid = publish_manifest(client)
    r = package(client, mid, make_zip(duplicate="C5"))
    assert r.status_code == 400
    assert any("同名文件" in e and "C5.json" in e for e in r.json()["errors"])


def test_subdirectory_and_junk_files_rejected(env):
    client, _, _ = env
    mid = publish_manifest(client)
    r = package(client, mid, make_zip(extra={"sub/C1.json": b"{}",
                                           "__MACOSX/._C1.json": b"x"}))
    assert r.status_code == 400
    errs = r.json()["errors"]
    assert any("sub/C1.json" in e for e in errs)
    assert any("__MACOSX/._C1.json" in e for e in errs)


def test_oversize_rejected(env, monkeypatch):
    client, _, _ = env
    import app.delivery as delivery
    monkeypatch.setattr(delivery, "MAX_EXTRACTED_BYTES", 64)
    mid = publish_manifest(client)
    big = json.dumps({"f": "x" * 200}).encode()
    r = package(client, mid, make_zip(content_fn=lambda _cid: big))
    assert r.status_code == 400
    assert any("8 MiB" in e or "上限" in e for e in r.json()["errors"])


def test_non_json_and_empty_rejected(env):
    client, _, _ = env
    mid = publish_manifest(client)
    r = package(client, mid, make_zip(
        content_fn=lambda cid: b"not-json" if cid == "C1" else (
            b"" if cid == "C2" else b"{}")))
    assert r.status_code == 400
    errs = r.json()["errors"]
    assert any("C1.json" in e and "JSON" in e for e in errs)
    assert any("C2.json" in e and "为空" in e for e in errs)


def test_not_a_zip_rejected(env):
    client, _, _ = env
    mid = publish_manifest(client)
    r = package(client, mid, b"this is not a zip")
    assert r.status_code == 400
    assert any("ZIP" in e for e in r.json()["errors"])


# --------------------------------------------------------------------------- #
# 重新打包：失败保旧、成功换旧
# --------------------------------------------------------------------------- #
def test_repackage_failure_keeps_previous_batch(env):
    client, _, _ = env
    mid = publish_manifest(client)
    b1 = package_ok(client, mid)

    # 重新打包失败（缺件）-> 整包拒收，上一对有效包仍可下载
    r = package(client, mid, make_zip(omit=("C1",)))
    assert r.status_code == 400
    for pkg in ("train", "eval"):
        assert client.get(f"/api/deliveries/{b1['batch_id']}/download/{pkg}").status_code == 200
    ds = client.get("/api/deliveries").json()["deliveries"]
    assert len(ds) == 1 and ds[0]["batch_id"] == b1["batch_id"] and ds[0]["downloadable"]


def test_repackage_success_invalidates_old_batch(env):
    client, _, _ = env
    mid = publish_manifest(client)
    b1 = package_ok(client, mid)
    b2 = package_ok(client, mid)
    assert b1["batch_id"] != b2["batch_id"]

    # 旧批次链接提示失效；新批次可下载
    for pkg in ("train", "eval"):
        r = client.get(f"/api/deliveries/{b1['batch_id']}/download/{pkg}")
        assert r.status_code == 409 and "失效" in r.json()["detail"]
        assert client.get(f"/api/deliveries/{b2['batch_id']}/download/{pkg}").status_code == 200
    ds = {b["batch_id"]: b for b in client.get("/api/deliveries").json()["deliveries"]}
    assert ds[b1["batch_id"]]["status"] == "已被新批次替换"
    assert ds[b2["batch_id"]]["status"] == "有效"


# --------------------------------------------------------------------------- #
# 清单失效：两包一律不得发布或下载
# --------------------------------------------------------------------------- #
def test_manifest_invalidation_blocks_publish_and_download(env):
    client, _, _ = env
    mid = publish_manifest(client)
    b1 = package_ok(client, mid)

    # 更正切分输入 -> 清单失效
    client.put("/api/locks", json={"locks": [{"case_id": "C1", "group": 2}]})

    for pkg in ("train", "eval"):
        r = client.get(f"/api/deliveries/{b1['batch_id']}/download/{pkg}")
        assert r.status_code == 409 and "清单已失效" in r.json()["detail"]
    r = package(client, mid, make_zip())
    assert r.status_code == 409 and "不得发布" in r.json()["detail"]
    ds = client.get("/api/deliveries").json()["deliveries"]
    assert ds[0]["status"] == "清单已失效" and ds[0]["downloadable"] is False


def test_commit_rechecks_manifest_when_input_changed_during_packaging(env):
    """打包期间输入被更正：提交阶段复核失败，上一对有效包不受影响。"""
    client, main, _ = env
    from app.db import StaleVersionError
    from app.delivery import build_packages

    mid = publish_manifest(client)
    b1 = package_ok(client, mid)

    manifest = client.get(f"/api/manifests/{mid}").json()
    features = {cid: b"{}" for cid in ALL_CASES}
    tmp_dir = main.DELIVERIES_DIR / "tmp-staged"
    pkgs = build_packages("staged-batch", manifest, features, tmp_dir)

    # “打包期间”：另一个窗口更正了切分输入
    client.put("/api/locks", json={"locks": [{"case_id": "C1", "group": 2}]})

    with pytest.raises(StaleVersionError):
        main.db.commit_delivery_batch(
            batch_id="staged-batch", manifest_id=mid,
            expected_version=manifest["dataset_version"],
            train=pkgs["train"], eval=pkgs["eval"])
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 旧批次因清单失效而不可得，但记录仍在、状态明确
    ds = client.get("/api/deliveries").json()["deliveries"]
    assert len(ds) == 1 and ds[0]["batch_id"] == b1["batch_id"]
    assert ds[0]["status"] == "清单已失效"


# --------------------------------------------------------------------------- #
# 两包必须同时存在才开放下载
# --------------------------------------------------------------------------- #
def test_missing_one_package_file_blocks_both_downloads(env):
    client, main, _ = env
    mid = publish_manifest(client)
    b1 = package_ok(client, mid)
    (main.DELIVERIES_DIR / b1["batch_id"] / "eval.zip").unlink()
    for pkg in ("train", "eval"):
        r = client.get(f"/api/deliveries/{b1['batch_id']}/download/{pkg}")
        assert r.status_code == 409 and "缺失" in r.json()["detail"]
    ds = client.get("/api/deliveries").json()["deliveries"]
    assert ds[0]["status"] == "交付文件缺失"


# --------------------------------------------------------------------------- #
# 重启持久化
# --------------------------------------------------------------------------- #
def test_persistence_across_restart(tmp_path, monkeypatch):
    db_file = tmp_path / "persist.db"
    monkeypatch.setenv("SPLIT_DB_PATH", str(db_file))
    import importlib
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c1:
        mid = publish_manifest(c1)
        b1 = package_ok(c1, mid)
        b1_id = b1["batch_id"]
        # 再发布失败场景：缺件 -> 拒收
        assert package(c1, mid, make_zip(omit=("C2",))).status_code == 400
    main.db.close()

    # 重启：批次记录与文件仍在，规则不变
    importlib.reload(main)
    with TestClient(main.app) as c2:
        ds = c2.get("/api/deliveries").json()["deliveries"]
        assert len(ds) == 1 and ds[0]["batch_id"] == b1_id and ds[0]["downloadable"]
        for pkg in ("train", "eval"):
            assert c2.get(f"/api/deliveries/{b1_id}/download/{pkg}").status_code == 200
        # 输入变动 -> 清单失效 -> 下载与发布都被拒
        c2.put("/api/locks", json={"locks": [{"case_id": "C1", "group": 2}]})
        assert c2.get(f"/api/deliveries/{b1_id}/download/train").status_code == 409
        assert package(c2, mid, make_zip()).status_code == 409
    main.db.close()

    # 再次重启：失效状态依旧
    importlib.reload(main)
    with TestClient(main.app) as c3:
        ds = c3.get("/api/deliveries").json()["deliveries"]
        assert ds[0]["status"] == "清单已失效"
        assert c3.get(f"/api/deliveries/{b1_id}/download/eval").status_code == 409
    main.db.close()


def test_startup_reconciles_orphan_dirs(tmp_path, monkeypatch):
    """崩溃遗留的临时目录与无记录孤儿目录在启动时被清理。"""
    db_file = tmp_path / "recon.db"
    monkeypatch.setenv("SPLIT_DB_PATH", str(db_file))
    import importlib
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c1:
        mid = publish_manifest(c1)
        b1 = package_ok(c1, mid)
        b1_id = b1["batch_id"]
        (main.DELIVERIES_DIR / "tmp-crashed").mkdir()
        (main.DELIVERIES_DIR / "orphan-batch").mkdir()
    main.db.close()

    importlib.reload(main)
    with TestClient(main.app) as c2:
        remaining = {p.name for p in main.DELIVERIES_DIR.iterdir()}
        assert remaining == {b1_id}
        assert c2.get(f"/api/deliveries/{b1_id}/download/train").status_code == 200
    main.db.close()


# --------------------------------------------------------------------------- #
# 并发打包同一清单：后者替换前者，状态一致
# --------------------------------------------------------------------------- #
def test_concurrent_packaging_same_manifest(env):
    client, _, _ = env
    mid = publish_manifest(client)
    import concurrent.futures

    def do_pack(_i):
        return package(client, mid, make_zip())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(do_pack, range(4)))
    assert all(r.status_code == 200 for r in results)
    ds = client.get("/api/deliveries").json()["deliveries"]
    assert len(ds) == 4
    downloadable = [b for b in ds if b["downloadable"]]
    assert len(downloadable) == 1  # 只有最后提交的批次可下载
    for b in ds:
        for pkg in ("train", "eval"):
            r = client.get(f"/api/deliveries/{b['batch_id']}/download/{pkg}")
            assert r.status_code == (200 if b["downloadable"] else 409)
