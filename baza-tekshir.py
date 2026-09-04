"""Graphify ChromaDB va HNSW bazasining sog'lig'ini tekshirish va tiklash vositasi.

Ishlatish:
    python baza-tekshir.py          # 5 bosqichli to'liq tekshiruv
    python baza-tekshir.py --tuzat  # Buzuq HNSW segmentini zaxira bilan xavfsiz chetga surish
"""

import argparse
import csv
import io
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime

# Windows konsolida UTF-8 chiqishini ta'minlash
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "kg_db"
COLLECTION_NAME = "graphify"
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

# Mos python interpretatorini aniqlash
RUN_PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# Agar tizim pythonidan chaqirilgan bo'lsa va chromadb import qila olmasa, avtomatik .venv orqali qayta yurgizish
try:
    import chromadb  # noqa: F401
except ImportError:
    if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON) and not any(arg.startswith("--_worker_") for arg in sys.argv):
        res = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
        sys.exit(res.returncode)

# Timeout chegaralari (soniyalarda)
TIMEOUT_SQLITE = 10.0
TIMEOUT_COLLECTION = 10.0
TIMEOUT_COUNT = 15.0
TIMEOUT_QUERY = 15.0
TIMEOUT_PROCS = 10.0

# Eski jarayon mezoni (soniyalarda: 1 soatdan oshgan jarayonlar)
STALE_PROCESS_THRESHOLD = 3600.0


def _worker_open_collection():
    """Worker: Kolleksiya ochilishini tekshirish."""
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_PATH))
    _ = client.get_collection(name=COLLECTION_NAME)
    if hasattr(client, "_system"):
        try:
            client._system.stop()
        except Exception:
            pass


def _worker_count():
    """Worker: Kolleksiya count() chaqiruvini bajarish (segfault himoyasi)."""
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_PATH))
    col = client.get_collection(name=COLLECTION_NAME)
    cnt = col.count()
    print(cnt)
    if hasattr(client, "_system"):
        try:
            client._system.stop()
        except Exception:
            pass


def _worker_query():
    """Worker: Kolleksiyadan sinov qidiruvini o'tkazish."""
    sys.path.insert(0, str(BASE_DIR))
    import db
    col = db.get_collection()
    res = col.query(query_texts=["sinov tekshiruvi"], n_results=1)
    ids = res.get("ids", [[]])
    count_found = len(ids[0]) if ids else 0
    print(count_found)


def get_server_processes():
    """Tizimdagi server.py jarayonlarini aniqlaydi."""
    procs = []
    # 1. wmic orqali aniqlash
    try:
        res = subprocess.run(
            ["wmic", "process", "where", "name='python.exe' or name='python3.exe'", "get", "ProcessId,ParentProcessId,CreationDate,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=TIMEOUT_PROCS
        )
        if res.returncode == 0 and res.stdout:
            reader = csv.reader(io.StringIO(res.stdout))
            rows = [r for r in reader if r and len(r) >= 5]
            for r in rows:
                cmd = r[1]
                created_str = r[2].strip()
                parent_pid = r[3].strip()
                pid = r[4].strip()
                if "server.py" in cmd and pid.isdigit():
                    procs.append({
                        "pid": int(pid),
                        "parent_pid": int(parent_pid) if parent_pid.isdigit() else 0,
                        "cmd": cmd,
                        "created_str": created_str,
                    })
            return procs
    except Exception:
        pass

    # 2. PowerShell zaxira usuli
    try:
        ps_cmd = 'Get-CimInstance Win32_Process | Where-Object { $_.Name -match "^python" -and $_.CommandLine -like "*server.py*" } | ForEach-Object { "$($_.ProcessId)|$($_.ParentProcessId)|$($_.CreationDate.ToString(\'yyyyMMddHHmmss\'))|$($_.CommandLine)" }'
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=TIMEOUT_PROCS)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = line.strip().split("|", 3)
                if len(parts) >= 4 and parts[0].isdigit():
                    procs.append({
                        "pid": int(parts[0]),
                        "parent_pid": int(parts[1]) if parts[1].isdigit() else 0,
                        "cmd": parts[3],
                        "created_str": parts[2],
                    })
    except Exception:
        pass

    return procs


def parse_creation_time(created_str: str):
    """wmic CreationDate formatini (YYYYMMDDHHMMSS...) datetime ga o'giradi."""
    if not created_str or len(created_str) < 14:
        return None
    try:
        dt_str = created_str[:14]
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except Exception:
        return None


def check_stale_processes(procs, threshold_seconds=STALE_PROCESS_THRESHOLD):
    """Eski (uzoq vaqt qolib ketgan) server.py jarayonlarini ajratadi."""
    now = datetime.now()
    stale = []
    for p in procs:
        c_time = parse_creation_time(p.get("created_str", ""))
        if c_time:
            age = (now - c_time).total_seconds()
            if age > threshold_seconds:
                stale.append({**p, "age_seconds": age})
    return stale


def run_worker_step(step_name: str, timeout: float):
    """Worker funksiyasini alohida jarayonda xavfsiz yurgizadi."""
    cmd = [RUN_PYTHON, str(Path(__file__).resolve()), f"--_worker_{step_name}"]
    t0 = time.monotonic()
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        return res.returncode, res.stdout.strip(), res.stderr.strip(), elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return -999, "", f"Timeout ({timeout}s dan oshdi)", elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        return -1, "", str(e), elapsed


def check_baza():
    """Baza holatini 5 bosqichda tekshiradi.

    Qaytaradi: (all_ok: bool)
    """
    print("=" * 65)
    print("Graphify Vektor Bazasining Sog'lig'ini Tekshirish")
    print(f"Baza manzili: {DB_PATH}")
    print(f"Kolleksiya:   {COLLECTION_NAME}")
    print("=" * 65)

    all_ok = True

    # -------------------------------------------------------------
    # 1-bosqich: SQLite ochiladimi, pragma quick_check o'tadimi, nechta qator
    # -------------------------------------------------------------
    t0 = time.monotonic()
    sqlite_ok = False
    row_count = 0
    sqlite_file = DB_PATH / "chroma.sqlite3"

    if not sqlite_file.exists():
        elapsed = time.monotonic() - t0
        print(f"❌ 1. SQLite tekshiruvi: chroma.sqlite3 fayli topilmadi ({elapsed:.2f}s)")
        all_ok = False
    else:
        try:
            conn = sqlite3.connect(str(sqlite_file), timeout=TIMEOUT_SQLITE)
            try:
                cur = conn.cursor()
                cur.execute("PRAGMA quick_check")
                qc_res = cur.fetchall()
                if qc_res and qc_res[0][0] == "ok":
                    cur.execute("SELECT count(*) FROM embeddings")
                    row_count = cur.fetchone()[0]
                    sqlite_ok = True
                else:
                    qc_text = str(qc_res)
            finally:
                conn.close()
            elapsed = time.monotonic() - t0
            if sqlite_ok:
                formatted_rows = f"{row_count:,}".replace(",", " ")
                print(f"✅ 1. SQLite butunligi: ok, {formatted_rows} qator ({elapsed:.2f}s)")
            else:
                print(f"❌ 1. SQLite tekshiruvi: pragma quick_check xatosi ({qc_text}) ({elapsed:.2f}s)")
                all_ok = False
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"❌ 1. SQLite tekshiruvi: {e} ({elapsed:.2f}s)")
            all_ok = False

    # -------------------------------------------------------------
    # 2-bosqich: Kolleksiya ochiladimi
    # -------------------------------------------------------------
    ret, out, err, elapsed = run_worker_step("open", TIMEOUT_COLLECTION)
    if ret == 0:
        print(f"✅ 2. Kolleksiya ochilishi: '{COLLECTION_NAME}' muvaffaqiyatli ochildi ({elapsed:.2f}s)")
    else:
        err_msg = err or out or f"Returncode: {ret}"
        print(f"❌ 2. Kolleksiya ochilishi: ochib bo'lmadi ({err_msg}) ({elapsed:.2f}s)")
        all_ok = False

    # -------------------------------------------------------------
    # 3-bosqich: count() alohida jarayonda chaqirilsin (segfault himoyasi)
    # -------------------------------------------------------------
    ret, out, err, elapsed = run_worker_step("count", TIMEOUT_COUNT)
    if ret == 0 and out.isdigit():
        chunk_count = int(out)
        formatted_chunks = f"{chunk_count:,}".replace(",", " ")
        print(f"✅ 3. Kolleksiya count(): {formatted_chunks} chunk (alohida jarayonda) ({elapsed:.2f}s)")
    elif ret == -999:
        print(f"❌ 3. Kolleksiya count(): Osilib qoldi / Timeout ({elapsed:.2f}s)")
        all_ok = False
    else:
        # Segfault kodlari: 139 (-11) yoki Windows 0xC0000005 (3221225477)
        print(f"❌ 3. Kolleksiya count(): Jarayon qulab tushdi (Segfault / Returncode: {ret}) ({elapsed:.2f}s)")
        if err:
            print(f"   Xatolik matni: {err}")
        all_ok = False

    # -------------------------------------------------------------
    # 4-bosqich: Sinov qidiruvi ishlaydimi
    # -------------------------------------------------------------
    ret, out, err, elapsed = run_worker_step("query", TIMEOUT_QUERY)
    if ret == 0:
        print(f"✅ 4. Sinov qidiruvi: HNSW vektor qidiruvi muvaffaqiyatli ({elapsed:.2f}s)")
    elif ret == -999:
        print(f"❌ 4. Sinov qidiruvi: Osilib qoldi / Timeout ({elapsed:.2f}s)")
        all_ok = False
    else:
        print(f"❌ 4. Sinov qidiruvi: Muvaffaqiyatsiz (kod: {ret}) ({elapsed:.2f}s)")
        if err:
            print(f"   Xatolik matni: {err}")
        all_ok = False

    # -------------------------------------------------------------
    # 5-bosqich: Nechta eski server.py jarayoni bor
    # -------------------------------------------------------------
    t0 = time.monotonic()
    procs = get_server_processes()
    stale_procs = check_stale_processes(procs, STALE_PROCESS_THRESHOLD)
    elapsed = time.monotonic() - t0

    if len(stale_procs) == 0:
        print(f"✅ 5. Eski server.py jarayonlari: 0 ta eski jarayon (jami: {len(procs)} ta faol) ({elapsed:.2f}s)")
    else:
        print(f"❌ 5. Eski server.py jarayonlari: {len(stale_procs)} ta eski jarayon aniqlandi ({elapsed:.2f}s)")
        for sp in stale_procs:
            print(f"   - PID {sp['pid']} (yoshi: {sp['age_seconds']/3600:.1f} soat)")
        all_ok = False

    print("=" * 65)
    if all_ok:
        print("Xulosa: Baza holati SOZ (barcha 5 ta bosqich muvaffaqiyatli o'tdi).")
    else:
        print("Xulosa: Bazada MUAMMO aniqlandi! Tiklash uchun: python baza-tekshir.py --tuzat")
    print("=" * 65)

    return all_ok


def get_current_vector_segment_id():
    """SQLite segments jadvalidan 'graphify' kolleksiyasining VECTOR segment UUID sini oladi."""
    sqlite_file = DB_PATH / "chroma.sqlite3"
    if not sqlite_file.exists():
        return None
    try:
        conn = sqlite3.connect(str(sqlite_file), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT s.id
                FROM segments s
                JOIN collections c ON s.collection = c.id
                WHERE c.name = ? AND s.scope = 'VECTOR'
            """, (COLLECTION_NAME,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
        finally:
            conn.close()
    except Exception:
        pass
    return None


def terminate_server_processes():
    """Tizimdagi server.py jarayonlarini xavfsiz to'xtatadi."""
    current_pid = os.getpid()
    procs = get_server_processes()
    terminated_count = 0
    for p in procs:
        pid = p["pid"]
        if pid == current_pid:
            continue
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
            terminated_count += 1
        except Exception:
            pass
    # Fayl qulflari to'liq bo'shashi uchun 1 soniya kutish
    time.sleep(1.0)
    return terminated_count


def tuzat_baza():
    """Buzilgan HNSW bazasini xavfsiz tiklash jarayoni.

    1. Avval zaxira — kg_db-zaxira-<sana-vaqt> (majburiy)
    2. Eski server.py jarayonlarini to'xtatish
    3. Buzilgan HNSW segment papkasini chetga surish (-buzuq qo'shimchasi bilan)
    4. Qayta sinash
    5. Ishlamasa — zaxiradan qaytarish va aniq xabar berish
    """
    print("=" * 65)
    print("Graphify Bazasini Xavfsiz Tiklash Jarayoni (--tuzat)")
    print("=" * 65)

    if not DB_PATH.exists():
        print(f"❌ XATO: Baza papkasi topilmadi ({DB_PATH}). Tiklash to'xtatildi.")
        sys.exit(1)

    # 1-qadam: Avval zaxira (majburiy, o'tkazib yuborilmaydi)
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = BASE_DIR / f"kg_db-zaxira-{now_str}"
    print(f"1-qadam: Baza zaxira nusxasi olinmoqda -> {backup_path.name}...")
    try:
        shutil.copytree(str(DB_PATH), str(backup_path))
        if not backup_path.exists() or not any(backup_path.iterdir()):
            raise RuntimeError("Zaxira papkasi bo'sh qoldi.")
        print(f"✅ 1. Majburiy zaxira muvaffaqiyatli olindi: {backup_path.name}")
    except Exception as e:
        print(f"⛔ XATO: Zaxira olib bo'lmadi ({e}).")
        print("Xavfsizlik qoidasiga ko'ra zaxirasiz hech qanday o'zgartirish MUMKIN EMAS! Jarayon to'xtatildi.")
        sys.exit(1)

    # 2-qadam: Eski server.py jarayonlarini to'xtatish
    print("2-qadam: Eski server.py jarayonlari to'xtatilmoqda...")
    terminated_count = terminate_server_processes()
    print(f"✅ 2. Server jarayonlari to'xtatildi ({terminated_count} ta jarayon to'xtatildi)")

    # 3-qadam: Buzilgan HNSW segment papkasini chetga surish
    print("3-qadam: Buzilgan HNSW segment papkasi aniqlanmoqda...")
    current_segment_id = get_current_vector_segment_id()
    uuid_regex = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

    target_segment_dir = None
    if current_segment_id:
        cand = DB_PATH / current_segment_id
        if cand.is_dir():
            target_segment_dir = cand

    if not target_segment_dir:
        # Agar SQLite dan olinmasa, mavjud UUID papkalar orasidan tanlash
        uuid_dirs = [d for d in DB_PATH.iterdir() if d.is_dir() and uuid_regex.match(d.name) and not d.name.endswith("-buzuq")]
        if uuid_dirs:
            uuid_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            target_segment_dir = uuid_dirs[0]

    if not target_segment_dir:
        print("⚠️ HNSW segment papkasi topilmadi yoki allaqachon chetga surilgan.")
    else:
        buzuq_name = f"{target_segment_dir.name}-buzuq"
        buzuq_path = DB_PATH / buzuq_name
        if buzuq_path.exists():
            buzuq_path = DB_PATH / f"{target_segment_dir.name}-buzuq-{now_str}"

        print(f"HNSW segment papkasi chetga surilmoqda: {target_segment_dir.name} -> {buzuq_path.name}")
        try:
            shutil.move(str(target_segment_dir), str(buzuq_path))
            print(f"✅ 3. Buzilgan HNSW segment chetga surildi: {buzuq_path.name}")
        except Exception as e:
            print(f"❌ Segment papkasini ko'chirib bo'lmadi: {e}")
            print("Zaxiradan tiklanmoqda...")
            shutil.rmtree(str(DB_PATH), ignore_errors=True)
            shutil.copytree(str(backup_path), str(DB_PATH))
            print(f"✅ Baza dastlabki holatiga qaytarildi: {backup_path.name}")
            sys.exit(1)

    # 4-qadam: Qayta sinash
    print("4-qadam: Baza holati qayta sinovdan o'tkazilmoqda...")
    ret_open, _, _, _ = run_worker_step("open", TIMEOUT_COLLECTION)
    ret_count, out_count, err_count, _ = run_worker_step("count", TIMEOUT_COUNT)

    test_ok = (ret_open == 0) and (ret_count == 0 and out_count.isdigit())

    if test_ok:
        print(f"✅ 4. Qayta sinov muvaffaqiyatli: baza ochildi va count() segfaultsiz ishladi ({out_count} chunk).")
        print("=" * 65)
        print("Xulosa: Tiklash muvaffaqiyatli bajarildi! Segfault xatosi bartaraf etildi.")
        print("⚠️ QAT'IY KO'RSATMA: HNSW vektor qidiruv indeksini tiklash uchun endi")
        print("   bazani qayta indekslash kerak (masalan: python indexer.py).")
        print("   Taqiq qoidasiga ko'ra avtomatik indekslash boshlanmadi.")
        print("=" * 65)
        sys.exit(0)
    else:
        # 5-qadam: Ishlamasa — zaxiradan qaytarish va aniq xabar berish
        print(f"❌ 4. Qayta sinov muvaffaqiyatsiz bo'ldi (open: {ret_open}, count: {ret_count}).")
        print(f"5-qadam: Baza olingan zaxiradan qaytarilmoqda ({backup_path.name})...")
        try:
            shutil.rmtree(str(DB_PATH), ignore_errors=True)
            shutil.copytree(str(backup_path), str(DB_PATH))
            print(f"✅ 5. Baza to'liq zaxira holatiga qaytarildi: {backup_path.name}")
        except Exception as e:
            print(f"❌ Zaxiradan qaytarishda xato: {e}")
        print("=" * 65)
        print("Xulosa: Avtomatik tiklash natija bermadi. Baza zaxira holatiga qaytarildi.")
        print("=" * 65)
        sys.exit(1)


def main():
    # Ichki worker chaqiruvlari uchun marshrutlash
    if len(sys.argv) > 1 and sys.argv[1].startswith("--_worker_"):
        worker_cmd = sys.argv[1][len("--_worker_"):]
        if worker_cmd == "open":
            _worker_open_collection()
            sys.exit(0)
        elif worker_cmd == "count":
            _worker_count()
            sys.exit(0)
        elif worker_cmd == "query":
            _worker_query()
            sys.exit(0)
        else:
            sys.exit(2)

    parser = argparse.ArgumentParser(description="Graphify bazasining sog'lig'ini tekshirish va tiklash vositasi")
    parser.add_argument("--tuzat", action="store_true", help="Buzilgan HNSW segmentini zaxira bilan xavfsiz chetga surish")
    args = parser.parse_args()

    if args.tuzat:
        tuzat_baza()
    else:
        ok = check_baza()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
