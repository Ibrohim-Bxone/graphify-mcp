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


_last_process_error: str | None = None


def get_last_process_error() -> str | None:
    """Return the last error reason encountered during process inspection."""
    return _last_process_error


def find_running_graphify_processes() -> list[dict] | None:
    """List running Graphify processes (python running server.py, server_http.py, bot).

    Returns:
        list[dict] of running processes if check succeeds (empty list if none found).
        None if check fails (timeout, non-zero return code, JSON error, non-Windows, etc.).
    """
    global _last_process_error
    _last_process_error = None

    if sys.platform != "win32":
        _last_process_error = "Windows bo'lmagan operatsion tizim"
        return None

    ps_cmd = (
        "@(Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like '*python*' } | "
        "Select-Object ProcessId, Name, CommandLine) | "
        "ConvertTo-Json -Compress"
    )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as e:
        _last_process_error = f"PowerShell timeout ({e.timeout}s)"
        return None
    except Exception as e:
        _last_process_error = f"PowerShell ishga tushirishda xatolik: {e}"
        return None

    if res.returncode != 0:
        err = res.stderr.strip() if res.stderr else f"rc={res.returncode}"
        _last_process_error = f"PowerShell xatolik kodi qaytardi ({err})"
        return None

    stdout = res.stdout.strip()
    if not stdout:
        _last_process_error = "PowerShell bo'sh chiqish qaytardi (JSON kutilgan edi)"
        return None

    try:
        data = json.loads(stdout)
    except Exception as e:
        _last_process_error = f"PowerShell JSON javobini o'qib bo'lmadi: {e}"
        return None

    if data is None:
        _last_process_error = "PowerShell bo'sh JSON (null) qaytardi"
        return None

    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        _last_process_error = f"PowerShell JSON ro'yxat emas: {type(data).__name__}"
        return None

    matches = []
    for item in data:
        if not isinstance(item, dict):
            _last_process_error = f"PowerShell jarayon elementi noto'g'ri (lug'at emas): {item!r}"
            return None
        cmd = item.get("CommandLine") or ""
        pid = item.get("ProcessId")
        name = item.get("Name") or "python.exe"
        if any(k in cmd.lower() for k in ["server.py", "server_http.py", "bot"]):
            matches.append({"pid": pid, "executable": name})

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

    from identity_utils import _check_identity_conflict, _parse_int, _parse_bool, _check_source_internal_conflict
    from embedders import ModelIdentityError

    try:
        _check_source_internal_conflict(col_meta, "Kolleksiya")
        _check_source_internal_conflict(manifest_identity, "Manifest")
    except ModelIdentityError as e:
        return {"status": "FAIL", "error": str(e)}

    keys_to_extract = [
        ("backend", str),
        ("model", str),
        ("dimension", _parse_int),
        ("model_digest", str),
        ("schema_version", _parse_int),
        ("distance", str),
        ("normalization", _parse_bool),
    ]

    resolved_c = {}
    for key, cast_type in keys_to_extract:
        val_plain = col_meta.get(key)
        val_pref = col_meta.get(f"graphify:{key}")
        if val_plain is not None and val_pref is not None:
            differ = False
            try:
                c1 = cast_type(val_plain)
                c2 = cast_type(val_pref)
                differ = (c1 != c2)
            except Exception:
                differ = True
            if differ:
                return {"status": "FAIL", "error": f"Kolleksiyada '{key}' va 'graphify:{key}' qiymatlari mos kelmadi: {val_plain} != {val_pref}"}
            resolved_c[key] = val_plain
        else:
            resolved_c[key] = val_plain if val_plain is not None else val_pref

    c_backend = resolved_c["backend"]
    c_model = resolved_c["model"]
    c_dim = resolved_c["dimension"]
    c_digest = resolved_c["model_digest"]
    c_schema = resolved_c["schema_version"]
    c_dist = resolved_c["distance"]
    c_norm = resolved_c["normalization"]

    if not c_backend:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'backend' identifikatsiyasi topilmadi"}
    if not c_model:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'model' identifikatsiyasi topilmadi"}
    if c_dim is None:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'dimension' identifikatsiyasi topilmadi"}
    if str(c_backend).lower() == "ollama" and not c_digest:
        return {"status": "FAIL", "error": "Baza kolleksiyasida Ollama uchun 'model_digest' topilmadi"}
    if c_schema is None:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'schema_version' topilmadi"}
    if not c_dist:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'distance' topilmadi"}
    if c_norm is None:
        return {"status": "FAIL", "error": "Baza kolleksiyasida 'normalization' topilmadi"}

    resolved_m = {}
    for key, cast_type in keys_to_extract:
        val_plain = manifest_identity.get(key)
        val_pref = manifest_identity.get(f"graphify:{key}")
        if val_plain is not None and val_pref is not None:
            differ = False
            try:
                c1 = cast_type(val_plain)
                c2 = cast_type(val_pref)
                differ = (c1 != c2)
            except Exception:
                differ = True
            if differ:
                return {"status": "FAIL", "error": f"Manifestda '{key}' va 'graphify:{key}' qiymatlari mos kelmadi: {val_plain} != {val_pref}"}
            resolved_m[key] = val_plain
        else:
            resolved_m[key] = val_plain if val_plain is not None else val_pref

    m_backend = resolved_m["backend"]
    m_model = resolved_m["model"]
    m_dim = resolved_m["dimension"]
    m_digest = resolved_m["model_digest"]
    m_schema = resolved_m["schema_version"]
    m_dist = resolved_m["distance"]
    m_norm = resolved_m["normalization"]

    if not m_backend or str(m_backend).lower() != str(c_backend).lower():
        return {"status": "FAIL", "error": f"Backend mos kelmadi: manifest={m_backend}, kolleksiya={c_backend}"}
    if not m_model or str(m_model) != str(c_model):
        return {"status": "FAIL", "error": f"Model nomi mos kelmadi: manifest={m_model}, kolleksiya={c_model}"}
    if m_dim is None:
        return {"status": "FAIL", "error": "Manifestda 'dimension' topilmadi"}

    try:
        m_dim_int = _parse_int(m_dim)
    except (ValueError, TypeError, OverflowError):
        return {"status": "FAIL", "error": f"Manifestda 'dimension' noto'g'ri formatda: {m_dim}"}

    try:
        c_dim_int = _parse_int(c_dim)
    except (ValueError, TypeError, OverflowError):
        return {"status": "FAIL", "error": f"Baza kolleksiyasida 'dimension' noto'g'ri formatda: {c_dim}"}

    if m_dim_int != c_dim_int:
        return {"status": "FAIL", "error": f"Dimension mos kelmadi: manifest={m_dim}, kolleksiya={c_dim}"}

    if str(c_backend).lower() == "ollama" and str(m_digest) != str(c_digest):
        return {"status": "FAIL", "error": f"Model digest mos kelmadi: manifest={m_digest}, kolleksiya={c_digest}"}

    if m_schema is None:
        return {"status": "FAIL", "error": "Manifestda 'schema_version' topilmadi"}

    try:
        m_schema_int = _parse_int(m_schema)
    except (ValueError, TypeError, OverflowError):
        return {"status": "FAIL", "error": f"Manifestda 'schema_version' noto'g'ri formatda: {m_schema}"}

    try:
        c_schema_int = _parse_int(c_schema)
    except (ValueError, TypeError, OverflowError):
        return {"status": "FAIL", "error": f"Baza kolleksiyasida 'schema_version' noto'g'ri formatda: {c_schema}"}

    if m_schema_int != c_schema_int:
        return {"status": "FAIL", "error": f"schema_version mos kelmadi: manifest={m_schema}, kolleksiya={c_schema}"}

    if not m_dist or str(m_dist).lower() != str(c_dist).lower():
        return {"status": "FAIL", "error": f"distance mos kelmadi: manifest={m_dist}, kolleksiya={c_dist}"}

    if m_norm is None:
        return {"status": "FAIL", "error": "Manifestda 'normalization' topilmadi"}

    try:
        norm_m = _parse_bool(m_norm)
    except (ValueError, TypeError, OverflowError):
        return {"status": "FAIL", "error": f"Manifestda 'normalization' noto'g'ri formatda: {m_norm}"}

    try:
        norm_c = _parse_bool(c_norm)
    except (ValueError, TypeError, OverflowError):
        return {"status": "FAIL", "error": f"Baza kolleksiyasida 'normalization' noto'g'ri formatda: {c_norm}"}

    if norm_m != norm_c:
        return {"status": "FAIL", "error": f"normalization mos kelmadi: manifest={m_norm}, kolleksiya={c_norm}"}

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


def check_v2_completeness(v2_dir: Path) -> dict:
    """Verify completeness of v2 database build.

    Checks:
    - build_complete.json exists (if missing -> FAIL "build tugallanmagan")
    - Model identity in build_complete.json matches graphify_manifest.json
    - v2 collection record count (collection.count()) equals manifest.jsonl line count.
    Opens ONLY the v2 database, never v1.
    """
    complete_file = v2_dir / "build_complete.json"
    if not complete_file.exists():
        return {
            "status": "FAIL",
            "error": "build tugallanmagan: build_complete.json topilmadi",
        }

    try:
        with open(complete_file, "r", encoding="utf-8") as f:
            complete_data = json.load(f)
    except Exception as e:
        return {
            "status": "FAIL",
            "error": f"build tugallanmagan (build_complete.json o'qishda xatolik: {e})",
        }

    if not isinstance(complete_data, dict):
        return {
            "status": "FAIL",
            "error": "build tugallanmagan (build_complete.json formati noto'g'ri)",
        }

    manifest_file = v2_dir / "graphify_manifest.json"
    if not manifest_file.exists():
        return {
            "status": "FAIL",
            "error": f"graphify_manifest.json topilmadi: {manifest_file}",
        }

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as e:
        return {
            "status": "FAIL",
            "error": f"Manifest o'qishda xatolik: {e}",
        }

    if not isinstance(manifest_data, dict):
        return {
            "status": "FAIL",
            "error": "graphify_manifest.json formati noto'g'ri (lug'at kutilgan edi)",
        }

    required_keys = ["model", "backend", "dimension", "schema_version", "distance", "normalization"]
    for k in required_keys:
        val = complete_data.get(k) if complete_data.get(k) is not None else complete_data.get(f"graphify:{k}")
        if val is None:
            return {
                "status": "FAIL",
                "error": f"build_complete.json da majburiy identity maydoni yetishmaydi: '{k}'",
            }
    complete_backend = complete_data.get("backend") if complete_data.get("backend") is not None else complete_data.get("graphify:backend")
    manifest_backend = manifest_data.get("backend") if manifest_data.get("backend") is not None else manifest_data.get("graphify:backend")
    has_manifest_digest = "model_digest" in manifest_data or "graphify:model_digest" in manifest_data
    has_complete_digest = "model_digest" in complete_data or "graphify:model_digest" in complete_data
    is_ollama = str(complete_backend or "").lower() == "ollama" or str(manifest_backend or "").lower() == "ollama"

    if is_ollama or has_manifest_digest or has_complete_digest:
        digest_val = complete_data.get("model_digest") if complete_data.get("model_digest") is not None else complete_data.get("graphify:model_digest")
        if digest_val is None:
            return {
                "status": "FAIL",
                "error": "build_complete.json da majburiy identity maydoni yetishmaydi: 'model_digest'",
            }

    from identity_utils import _check_source_internal_conflict, _check_identity_conflict
    from embedders import ModelIdentityError

    try:
        _check_source_internal_conflict(complete_data, "build_complete")
        _check_source_internal_conflict(manifest_data, "Manifest")
        _check_identity_conflict(complete_data, "build_complete", manifest_data, "manifest")
    except ModelIdentityError as e:
        return {
            "status": "FAIL",
            "error": f"build_complete.json va graphify_manifest.json mos kelmadi: {e}",
        }

    manifest_l_path = v2_dir / "manifest.jsonl"
    if not manifest_l_path.exists():
        return {
            "status": "FAIL",
            "error": f"manifest.jsonl topilmadi: {manifest_l_path}",
        }

    manifest_lines = 0
    try:
        with open(manifest_l_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    manifest_lines += 1
    except Exception as e:
        return {
            "status": "FAIL",
            "error": f"manifest.jsonl o'qishda xatolik: {e}",
        }

    import gc
    import tempfile
    import shutil
    import chromadb

    col_count = None
    client_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="graphify-v2-check-", ignore_cleanup_errors=True) as tmp_check_dir:
            tmp_check_path = Path(tmp_check_dir)
            sqlite_src = v2_dir / "chroma.sqlite3"
            if sqlite_src.exists():
                shutil.copy2(sqlite_src, tmp_check_path / "chroma.sqlite3")
                for extra in ("chroma.sqlite3-wal", "chroma.sqlite3-shm"):
                    if (v2_dir / extra).exists():
                        shutil.copy2(v2_dir / extra, tmp_check_path / extra)

            client = None
            col = None
            try:
                try:
                    client = chromadb.PersistentClient(path=str(tmp_check_path))
                    col = client.get_collection("graphify")
                    col_count = col.count()
                finally:
                    try:
                        if client is not None:
                            if hasattr(client, "close") and callable(client.close):
                                client.close()
                            elif hasattr(client, "_system") and hasattr(client._system, "stop") and callable(client._system.stop):
                                client._system.stop()
                    finally:
                        del client, col
                        gc.collect()
            except Exception as e:
                client_error = e
    except Exception as e:
        if col_count is None or client_error is not None:
            err = client_error or e
            return {
                "status": "FAIL",
                "error": f"v2 kolleksiyasini ochishda xatolik: {err}",
            }

    if client_error is not None:
        return {
            "status": "FAIL",
            "error": f"v2 kolleksiyasini ochishda xatolik: {client_error}",
        }

    if col_count is None:
        return {
            "status": "FAIL",
            "error": "v2 kolleksiyasini ochishda xatolik",
        }

    if col_count != manifest_lines:
        return {
            "status": "FAIL",
            "error": f"v2 kolleksiya yozuv soni ({col_count}) manifest.jsonl qatorlar soni ({manifest_lines}) bilan teng emas",
        }

    return {
        "status": "PASS",
        "count": col_count,
        "manifest_lines": manifest_lines,
    }


def check_v2_normalization(v2_dir: Path) -> dict:
    """Verify v2 database embedding normalization in a temporary copy following existing style.

    - If normalization: True, verifies a sample of up to 20 vectors has L2 norm == 1.0 (within 1e-3).
      If norm violates 1±1e-3, returns FAIL.
    - If normalization: False, returns WARN ("vektorlar normallashtirilmagan — reyting normaga og'adi").
    """
    manifest_file = v2_dir / "graphify_manifest.json"
    norm_val = None
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            if isinstance(m_data, dict):
                norm_val = m_data.get("normalization")
                if norm_val is None:
                    norm_val = m_data.get("graphify:normalization")
        except Exception as e:
            return {"status": "FAIL", "error": f"graphify_manifest.json o'qishda xatolik: {e}"}

    db_file = v2_dir / "chroma.sqlite3"
    if not db_file.exists():
        return {"status": "FAIL", "error": f"Chroma DB fayli topilmadi: {db_file}"}

    if norm_val is None:
        try:
            conn = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True, timeout=5.0)
            try:
                cur = conn.cursor()
                cur.execute("SELECT id FROM collections WHERE name = 'graphify'")
                c_row = cur.fetchone()
                if c_row:
                    cur.execute(
                        "SELECT key, bool_value, str_value, int_value FROM collection_metadata WHERE collection_id = ?",
                        (c_row[0],),
                    )
                    for k, b, s, i in cur.fetchall():
                        if k in ("normalization", "graphify:normalization"):
                            norm_val = bool(b) if b is not None else s if s is not None else bool(i)
                            break
            finally:
                conn.close()
        except Exception as e:
            return {"status": "FAIL", "error": f"Baza metadata o'qishda xatolik: {e}"}

    if norm_val is None:
        return {"status": "FAIL", "error": "Baza identity'sida 'normalization' topilmadi"}

    from identity_utils import _parse_bool
    try:
        norm_bool = _parse_bool(norm_val)
    except Exception as e:
        return {"status": "FAIL", "error": f"Baza identity'sida 'normalization' noto'g'ri: {e}"}

    if not norm_bool:
        return {
            "status": "WARN",
            "message": "vektorlar normallashtirilmagan — reyting normaga og'adi",
        }

    # normalization: True -> check sample of up to 20 records in tmp copy
    import tempfile
    import shutil
    import gc
    import chromadb
    import numpy as np

    client = None
    col = None
    try:
        with tempfile.TemporaryDirectory(prefix="graphify-v2-norm-", ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            sqlite_src = v2_dir / "chroma.sqlite3"
            if sqlite_src.exists():
                shutil.copy2(sqlite_src, tmp_path / "chroma.sqlite3")
                for extra in ("chroma.sqlite3-wal", "chroma.sqlite3-shm"):
                    if (v2_dir / extra).exists():
                        shutil.copy2(v2_dir / extra, tmp_path / extra)
            for item in v2_dir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    shutil.copytree(item, tmp_path / item.name, dirs_exist_ok=True)

            try:
                client = chromadb.PersistentClient(path=str(tmp_path))
                col = client.get_collection("graphify")
                cnt = col.count()
                if cnt == 0:
                    return {"status": "FAIL", "error": "Kolleksiya bo'sh (yozuvlar yo'q)"}
                sample_limit = min(20, cnt)
                res = col.get(limit=sample_limit, include=["embeddings"])
                embs = res.get("embeddings")
                if embs is None or len(embs) == 0:
                    return {"status": "FAIL", "error": "Embeddinglar topilmadi"}

                for idx, vec in enumerate(embs):
                    if vec is None or len(vec) == 0:
                        return {"status": "FAIL", "error": f"Yozuv {idx} da embedding yo'q"}
                    v = np.array(vec, dtype=np.float32)
                    norm = float(np.linalg.norm(v))
                    if not np.isfinite(norm) or not np.all(np.isfinite(v)) or abs(norm - 1.0) > 1e-3:
                        return {
                            "status": "FAIL",
                            "error": f"Namunaviy yozuv normasi 1±1e-3 oralig'ida emas: norm={norm:.6f} (kutilgan=1.0)",
                        }
                return {"status": "PASS", "sample_count": len(embs)}
            finally:
                try:
                    if client is not None:
                        if hasattr(client, "close") and callable(client.close):
                            client.close()
                        elif hasattr(client, "_system") and hasattr(client._system, "stop"):
                            client._system.stop()
                finally:
                    del client, col
                    gc.collect()
    except Exception as e:
        return {"status": "FAIL", "error": f"Namunaviy vektorlarni tekshirishda xatolik: {e}"}


def print_checklist(
    processes: list[dict] | None,
    drift: dict,
    identity: dict,
    launch_configs: list[dict],
    v1_dir: Path,
    v2_dir: Path,
    completeness: dict | None = None,
    normalization: dict | None = None,
) -> bool:
    """Print formatted pre-switch checklist and ordered execution steps.

    Returns True if system is ready to switch, False otherwise.
    """
    if completeness is None and v2_dir is not None and (v2_dir / "graphify_manifest.json").exists():
        completeness = check_v2_completeness(v2_dir)

    if normalization is None and v2_dir is not None and (v2_dir / "graphify_manifest.json").exists():
        normalization = check_v2_normalization(v2_dir)

    print("=" * 70)
    print("[CHECKLIST] GRAPHIFY v2 O'TISH OLDIDAN TEKSHIRUV (SWITCH CHECKLIST)")
    print("   (Read-only rejim: hech qanday o'zgartirish kiritilmadi)")
    print("=" * 70)

    # 1. Running processes
    print("\n1. ISHLAB TURGAN GRAPHIFY JARAYONLARI:")
    if processes is None:
        err = get_last_process_error() or "jarayonlarni tekshirib bo'lmadi"
        print(f"   [FAIL] Yozuvchi jarayonlarni tekshirib bo'lmadi: {err}")
    elif processes:
        print(f"   [WARN] Topildi: {len(processes)} ta faol jarayon:")
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
        print("   [PASS] Faol Graphify yozuvchi jarayonlari topilmadi.")

    # 2. Drift
    print("\n2. v1 VA v2 MANIFEST O'RTASIDAGI FARQ (DRIFT):")
    if drift.get("status") == "OK":
        td = drift.get("total_drift", 0)
        if td == 0:
            print(f"   [PASS] Drift = 0 (v1 va v2 to'liq sinxron, jami {drift.get('total_v1', 0)} ta yozuv).")
        else:
            print(f"   [WARN] Drift mavjud: {td} ta o'zgarish!")
            print(f"      - v1 dagi yangi: {drift.get('new_in_v1', 0)} ta")
            print(f"      - v1 da o'zgargan: {drift.get('modified_in_v1', 0)} ta")
            print(f"      - v1 da o'chirilgan: {drift.get('deleted_in_v1', 0)} ta")
            print("   -> O'tishdan oldin oxirgi sinxronizatsiya bajarilishi shart.")
    else:
        print(f"   [FAIL] Drift tekshiruvi: {drift.get('message')}")

    # 3. Model Identity, Completeness & Normalization
    print("\n3. v2 MODEL IDENTIFIKATSIYASI VA BAZA TO'LIQLIGI (SELF-TEST):")
    if identity.get("status") == "PASS":
        print("   [PASS] Self-test: PASS")
        print(f"      - Model: {identity.get('model')} ({identity.get('dimension')}-d)")
        print(f"      - Backend: {identity.get('backend')}")
        digest = identity.get("digest")
        digest_str = f"{digest[:24]}..." if digest else "-"
        print(f"      - Digest: {digest_str}")
    else:
        print(f"   [FAIL] Self-test: FAIL ({identity.get('error')})")

    if completeness is not None:
        if completeness.get("status") == "PASS":
            print(f"   [PASS] Baza to'liqligi: PASS ({completeness.get('count', 0)} ta yozuv)")
        else:
            print(f"   [FAIL] Baza to'liqligi: FAIL ({completeness.get('error')})")

    if normalization is not None:
        if normalization.get("status") == "PASS":
            print(f"   [PASS] Vektorlar normalizatsiyasi: PASS ({normalization.get('sample_count', 20)} ta namunaviy vektor normasi 1±1e-3)")
        elif normalization.get("status") == "WARN":
            warn_msg = normalization.get("message") or "vektorlar normallashtirilmagan — reyting normaga og'adi"
            print(f"   [WARN] {warn_msg}")
        else:
            print(f"   [FAIL] Vektorlar normalizatsiyasi: FAIL ({normalization.get('error')})")

    # 4. Launch Configs
    print("\n4. SOZLAMA FAYLLARI (Faqat kalit nomi tekshirildi, qiymatlar yashirin):")
    for lc in launch_configs:
        status_icon = "SET" if lc["has_graphify_db_path"] else ("YO'Q" if lc["exists"] else "TOPILMADI")
        print(f"   - {lc['path']}: GRAPHIFY_DB_PATH = [{status_icon}]")

    # 5. Ordered steps
    print("\n" + "=" * 70)
    print("[QADAMLAR] TARTIBLANGAN O'TISH BOSQICHLARI (ORDERED SWITCH STEPS):")
    print("=" * 70)
    print("1. Barcha yozuvchi xizmatlarni to'xtatish (MCP, HTTP server, Telegram bot).")
    print(f"2. Yakuniy muzlatish va sinxronizatsiya: python migrate_v2/build_v2.py --sync --v1-db {v1_dir} --out {v2_dir}")
    print("3. Baza butunligini to'liq tekshirish (100% yozuvlar va keyword_index tasdig'i).")
    print(f"4. Barcha sozlama fayllarida GRAPHIFY_DB_PATH ni {v2_dir.resolve()} ga yangilash.")
    print("5. Xizmatlarni qayta ishga tushirish (MCP, HTTP, Bot).")
    print("6. Jonli tizimda 1 ta test qidiruv so'rovi bilan v2 model identifikatsiyasini tekshirish.")
    print("7. Yozish rejimini qayta ochish.\n")

    print("=" * 70)
    print("[ROLLBACK] ORQAGA QAYTISH BOSQICHLARI (ORDERED ROLLBACK STEPS):")
    print("=" * 70)
    print("1. v2 dagi yozuvchi jarayonlarni zudlik bilan to'xtatish.")
    print("2. v2 da yaratilgan yangi yozuvlarni v1 ga yarashtirish (reconciliation).")
    print(f"3. Sozlama fayllarida GRAPHIFY_DB_PATH ni {v1_dir.resolve()} ga qaytarish.")
    print("4. models_config.json ni olib tashlash yoki lokal rejimga qaytarish.")
    print("5. Xizmatlarni qayta ishga tushirish.")
    print("6. v1 ishini tekshirib, yozishni qayta ochish.\n")

    is_ready = (
        processes is not None
        and len(processes) == 0
        and drift.get("status") == "OK"
        and drift.get("total_drift", 0) == 0
        and identity.get("status") == "PASS"
        and (completeness is None or completeness.get("status") == "PASS")
        and (normalization is None or normalization.get("status") in ("PASS", "WARN"))
    )

    print("=" * 70)
    if is_ready:
        print("XULOSA: [PASS] Tizim v2 ga o'tishga tayyor.")
    else:
        print("XULOSA: [FAIL] Tizim v2 ga o'tishga tayyor emas.")
    print("=" * 70)

    return is_ready


def main():
    parser = argparse.ArgumentParser(description="Pre-switch checklist for Graphify v2")
    parser.add_argument("--v1-db", default="kg_db", help="Path to v1 database (default: kg_db)")
    parser.add_argument("--v2-db", default="kg_db_v2b", help="Path to v2 database (default: kg_db_v2b)")
    args = parser.parse_args()

    v1_dir = Path(args.v1_db)
    v2_dir = Path(args.v2_db)

    processes = find_running_graphify_processes()
    drift = check_v1_manifest_drift(v1_dir, v2_dir)
    identity = check_v2_identity(v2_dir)
    completeness = check_v2_completeness(v2_dir)
    normalization = check_v2_normalization(v2_dir)
    launch_configs = inspect_launch_configs()

    is_ready = print_checklist(
        processes, drift, identity, launch_configs, v1_dir, v2_dir,
        completeness=completeness, normalization=normalization
    )
    if not is_ready:
        sys.exit(1)


if __name__ == "__main__":
    main()
