#!/usr/bin/env python3
"""Read-only pre-switch checklist for Graphify v2 migration.

Inspects running Graphify processes, checks v1 vs manifest drift counts,
runs v2 model identity self-test, inspects launch configs (keys by NAME only, never values),
and prints exact ordered switch and rollback steps.
Performs NO switch and modifies NO files or processes.
"""

import argparse
import json
import os
import sqlite3
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from migrate_v2.build_v2 import read_v1_records, compute_hashes


def find_running_graphify_processes() -> list[dict]:
    """List running Graphify processes (python running server.py, server_http.py, bot)."""
    matches = []
    # Try PowerShell CimInstance on Windows
    if sys.platform == "win32":
        ps_cmd = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -like '*python*' } | "
            "Select-Object ProcessId, Name, CommandLine | "
            "ConvertTo-Json -Compress"
        )
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0 and res.stdout.strip():
                try:
                    data = json.loads(res.stdout)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        cmd = item.get("CommandLine") or ""
                        pid = item.get("ProcessId")
                        name = item.get("Name") or "python.exe"
                        if any(k in cmd.lower() for k in ["server.py", "server_http.py", "bot"]):
                            matches.append({"pid": pid, "executable": name})
                except Exception:
                    pass
        except Exception:
            pass

    return matches


def check_v1_manifest_drift(v1_dir: Path, v2_dir: Path) -> dict:
    """Compare v1 SQLite database against v2 manifest.jsonl."""
    manifest_path = v2_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {
            "status": "NO_MANIFEST",
            "message": f"Manifest fayli topilmadi: {manifest_path}",
            "new_in_v1": 0,
            "modified_in_v1": 0,
            "deleted_in_v1": 0,
            "total_drift": 0,
        }

    manifest = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    manifest[item["id"]] = item
                except Exception:
                    pass

    try:
        col_name, col_meta, col_config, v1_mtime, v1_records = read_v1_records(v1_dir)
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"v1 o'qishda xatolik: {e}",
            "new_in_v1": 0,
            "modified_in_v1": 0,
            "deleted_in_v1": 0,
            "total_drift": 0,
        }

    v1_map = {r["id"]: r for r in v1_records}
    new_in_v1 = 0
    modified_in_v1 = 0

    for r in v1_records:
        rid = r["id"]
        d_hash, m_hash = compute_hashes(r["document"], r["metadata"])
        if rid not in manifest:
            new_in_v1 += 1
        elif manifest[rid].get("doc_sha256") != d_hash or manifest[rid].get("meta_sha256") != m_hash:
            modified_in_v1 += 1

    deleted_in_v1 = sum(1 for mid in manifest if mid not in v1_map)
    total_drift = new_in_v1 + modified_in_v1 + deleted_in_v1

    return {
        "status": "OK",
        "total_v1": len(v1_records),
        "total_manifest": len(manifest),
        "new_in_v1": new_in_v1,
        "modified_in_v1": modified_in_v1,
        "deleted_in_v1": deleted_in_v1,
        "total_drift": total_drift,
    }


def check_v2_identity(v2_dir: Path) -> dict:
    """Self-test v2 database identity: requires DB file and compares persisted collection identity with manifest."""
    manifest_file = v2_dir / "graphify_manifest.json"
    if not manifest_file.exists():
        return {
            "status": "FAIL",
            "error": f"graphify_manifest.json topilmadi: {manifest_file}",
        }

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_identity = json.load(f)
    except Exception as e:
        return {"status": "FAIL", "error": f"Manifest o'qishda xatolik: {e}"}

    if not isinstance(manifest_identity, dict):
        return {"status": "FAIL", "error": "Manifest fayli yaroqsiz (JSON lug'at emas)"}

    db_file = v2_dir / "chroma.sqlite3"
    if not db_file.exists():
        return {
            "status": "FAIL",
            "error": f"Chroma ma'lumotlar bazasi fayli topilmadi: {db_file}",
        }

    try:
        conn = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True, timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM collections")
            col_rows = cur.fetchall()
            if not col_rows:
                return {"status": "FAIL", "error": f"Bazada birorta ham kolleksiya topilmadi: {db_file}"}

            target_id = None
            for cid, cname in col_rows:
                if cname == "graphify":
                    target_id = cid
                    break
            if not target_id:
                return {"status": "FAIL", "error": f"Bazada 'graphify' kolleksiyasi topilmadi: {db_file}"}

            cur.execute("""
                SELECT key, str_value, int_value, float_value, bool_value
                FROM collection_metadata
                WHERE collection_id = ?
            """, (target_id,))
            rows = cur.fetchall()
            col_meta = {}
            for k, s, i, f, b in rows:
                val = bool(b) if b is not None else i if i is not None else f if f is not None else s
                col_meta[k] = val
        finally:
            conn.close()
    except Exception as e:
        return {"status": "FAIL", "error": f"Baza kolleksiyasini o'qishda xatolik: {e}"}

    from db import _check_identity_conflict
    from embedders import ModelIdentityError

    c_backend = col_meta.get("backend") or col_meta.get("graphify:backend")
    c_model = col_meta.get("model") or col_meta.get("graphify:model")
    c_dim = col_meta.get("dimension") or col_meta.get("graphify:dimension")
    c_digest = col_meta.get("model_digest") or col_meta.get("graphify:model_digest")

    if not c_backend:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'backend' identifikatsiyasi topilmadi"}
    if not c_model:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'model' identifikatsiyasi topilmadi"}
    if c_dim is None:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'dimension' identifikatsiyasi topilmadi"}
    if str(c_backend).lower() == "ollama" and not c_digest:
        return {"status": "FAIL", "error": "Baza kolleksiyasida Ollama uchun 'model_digest' topilmadi"}

    m_backend = manifest_identity.get("backend")
    m_model = manifest_identity.get("model")
    m_dim = manifest_identity.get("dimension")
    m_digest = manifest_identity.get("model_digest")

    if not m_backend or str(m_backend).lower() != str(c_backend).lower():
        return {"status": "FAIL", "error": f"Backend mos kelmadi: manifest={m_backend}, kolleksiya={c_backend}"}
    if not m_model or str(m_model) != str(c_model):
        return {"status": "FAIL", "error": f"Model nomi mos kelmadi: manifest={m_model}, kolleksiya={c_model}"}
    if m_dim is None or int(m_dim) != int(c_dim):
        return {"status": "FAIL", "error": f"Dimension mos kelmadi: manifest={m_dim}, kolleksiya={c_dim}"}
    if str(c_backend).lower() == "ollama" and str(m_digest) != str(c_digest):
        return {"status": "FAIL", "error": f"Model digest mos kelmadi: manifest={m_digest}, kolleksiya={c_digest}"}

    try:
        _check_identity_conflict(manifest_identity, "manifest", col_meta, "kolleksiya")
    except ModelIdentityError as e:
        return {"status": "FAIL", "error": str(e)}

    from embedders import build_identity_embedder

    try:
        embedder = build_identity_embedder(manifest_identity)
        test_vec = embedder.encode(["self-test"])
        dim = len(test_vec[0]) if hasattr(test_vec[0], "__len__") else len(test_vec)
        return {
            "status": "PASS",
            "model": manifest_identity.get("model"),
            "backend": manifest_identity.get("backend"),
            "dimension": dim,
            "digest": manifest_identity.get("model_digest"),
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


def inspect_launch_configs() -> list[dict]:
    """Inspect launch configs for GRAPHIFY_DB_PATH setting (keys by NAME only, never values)."""
    home = Path.home()
    configs_to_check = [
        home / ".claude.json",
        home / ".codex" / "config.toml",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / "bot" / ".env",
        PROJECT_ROOT.parent / "kanal-konveyer" / ".env",
    ]

    results = []
    for cfg in configs_to_check:
        if not cfg.exists():
            results.append({"path": str(cfg), "exists": False, "has_graphify_db_path": False})
            continue

        has_key = False
        try:
            content = cfg.read_text(encoding="utf-8", errors="replace")
            # Search by key name only: GRAPHIFY_DB_PATH
            if "GRAPHIFY_DB_PATH" in content:
                has_key = True
        except Exception:
            pass

        results.append({
            "path": str(cfg),
            "exists": True,
            "has_graphify_db_path": has_key,
        })
    return results


def print_checklist(
    processes: list[dict], drift: dict, identity: dict, launch_configs: list[dict], v1_dir: Path, v2_dir: Path
) -> None:
    """Print formatted pre-switch checklist and ordered execution steps."""
    print("=" * 70)
    print("📋 GRAPHIFY v2 O'TISH OLDIDAN TEKSHIRUV (SWITCH CHECKLIST)")
    print("   (Read-only rejim: hech qanday o'zgartirish kiritilmadi)")
    print("=" * 70)

    # 1. Running processes
    print("\n1. ISHLAB TURGAN GRAPHIFY JARAYONLARI:")
    if processes:
        print(f"   ??  Topildi: {len(processes)} ta faol jarayon:")
        for p in processes:
            exe = p.get("executable")
            if not exe and "command" in p:
                raw_cmd = p["command"].strip()
                if raw_cmd.startswith('"'):
                    exe = raw_cmd[1:].split('"', 1)[0]
                else:
                    exe = raw_cmd.split()[0]
                exe = Path(exe).name
            if not exe:
                exe = "python"
            print(f"      - PID {p['pid']}: {exe}")
        print("   -> O'tishdan oldin ushbu yozuvchi jarayonlar to'xtatilishi shart.")
    else:
        print("   ? Faol Graphify yozuvchi jarayonlari topilmadi.")

    # 2. Drift
    print("\n2. v1 VA v2 MANIFEST O'RTASIDAGI FARQ (DRIFT):")
    if drift.get("status") == "OK":
        td = drift["total_drift"]
        if td == 0:
            print(f"   ? Drift = 0 (v1 va v2 to'liq sinxron, jami {drift['total_v1']} ta yozuv).")
        else:
            print(f"   ??  Drift mavjud: {td} ta o'zgarish!")
            print(f"      - v1 dagi yangi: {drift['new_in_v1']} ta")
            print(f"      - v1 da o'zgargan: {drift['modified_in_v1']} ta")
            print(f"      - v1 da o'chirilgan: {drift['deleted_in_v1']} ta")
            print("   -> O'tishdan oldin oxirgi sinxronizatsiya bajarilishi shart.")
    else:
        print(f"   ??  Drift tekshiruvi: {drift.get('message')}")

    # 3. Model Identity
    print("\n3. v2 MODEL IDENTIFIKATSIYASI VA SELF-TEST:")
    if identity.get("status") == "PASS":
        print(f"   ? Self-test: PASS")
        print(f"      - Model: {identity.get('model')} ({identity.get('dimension')}-d)")
        print(f"      - Backend: {identity.get('backend')}")
        print(f"      - Digest: {identity.get('digest')[:24]}...")
    else:
        print(f"   ? Self-test: FAIL ({identity.get('error')})")

    # 4. Launch Configs
    print("\n4. SOZLAMA FAYLLARI (Faqat kalit nomi tekshirildi, qiymatlar yashirin):")
    for lc in launch_configs:
        status_icon = "SET" if lc["has_graphify_db_path"] else ("YO'Q" if lc["exists"] else "TOPILMADI")
        print(f"   - {lc['path']}: GRAPHIFY_DB_PATH = [{status_icon}]")

    # 5. Ordered steps
    print("\n" + "=" * 70)
    print("?? TARTIBLANGAN O'TISH BOSQICHLARI (ORDERED SWITCH STEPS):")
    print("=" * 70)
    print("1. Barcha yozuvchi xizmatlarni to'xtatish (MCP, HTTP server, Telegram bot).")
    print(f"2. Yakuniy muzlatish va sinxronizatsiya: python migrate_v2/build_v2.py --sync --v1-db {v1_dir} --out {v2_dir}")
    print("3. Baza butunligini to'liq tekshirish (100% yozuvlar va keyword_index tasdig'i).")
    print(f"4. Barcha sozlama fayllarida GRAPHIFY_DB_PATH ni {v2_dir.resolve()} ga yangilash.")
    print("5. Xizmatlarni qayta ishga tushirish (MCP, HTTP, Bot).")
    print("6. Jonli tizimda 1 ta test qidiruv so'rovi bilan v2 model identifikatsiyasini tekshirish.")
    print("7. Yozish rejimini qayta ochish.\n")

    print("=" * 70)
    print("?? ORQAGA QAYTISH BOSQICHLARI (ORDERED ROLLBACK STEPS):")
    print("=" * 70)
    print("1. v2 dagi yozuvchi jarayonlarni zudlik bilan to'xtatish.")
    print("2. v2 da yaratilgan yangi yozuvlarni v1 ga yarashtirish (reconciliation).")
    print(f"3. Sozlama fayllarida GRAPHIFY_DB_PATH ni {v1_dir.resolve()} ga qaytarish.")
    print("4. models_config.json ni olib tashlash yoki lokal rejimga qaytarish.")
    print("5. Xizmatlarni qayta ishga tushirish.")
    print("6. v1 ishini tekshirib, yozishni qayta ochish.\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-switch checklist for Graphify v2")
    parser.add_argument("--v1-db", default="kg_db", help="Path to v1 database (default: kg_db)")
    parser.add_argument("--v2-db", default="kg_db_v2", help="Path to v2 database (default: kg_db_v2)")
    args = parser.parse_args()

    v1_dir = Path(args.v1_db)
    v2_dir = Path(args.v2_db)

    processes = find_running_graphify_processes()
    drift = check_v1_manifest_drift(v1_dir, v2_dir)
    identity = check_v2_identity(v2_dir)
    launch_configs = inspect_launch_configs()

    print_checklist(processes, drift, identity, launch_configs, v1_dir, v2_dir)


if __name__ == "__main__":
    main()
