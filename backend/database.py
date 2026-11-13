"""
Database SQLite untuk menyimpan riwayat prediksi klasifikasi.

SQLite dipilih karena ringan, tanpa server terpisah, dan cukup untuk
mencatat riwayat prediksi website. File database otomatis dibuat di
backend/data/predictions.db saat pertama kali dijalankan.

Jika nanti ingin skala lebih besar / multi-user, bagian ini bisa diganti
ke PostgreSQL (mis. Neon) dengan menyesuaikan koneksi di fungsi get_conn().
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.getenv("DB_PATH", os.path.join(DB_DIR, "predictions.db"))


def get_conn() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Buat tabel riwayat prediksi jika belum ada."""
    conn = get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT    NOT NULL,
                predicted_class TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                prob_normal     REAL    NOT NULL,
                prob_luad       REAL    NOT NULL,
                prob_lusc       REAL    NOT NULL,
                created_at      TEXT    NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_prediction(filename: str, result: Dict[str, Any]) -> int:
    """Simpan satu hasil prediksi, kembalikan id baris."""
    probs = result["probabilities"]
    conn = get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (filename, predicted_class, confidence,
                 prob_normal, prob_luad, prob_lusc, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                result["predicted_class"],
                float(result["confidence"]),
                float(probs.get("Normal", 0.0)),
                float(probs.get("LUAD", 0.0)),
                float(probs.get("LUSC", 0.0)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Ambil riwayat prediksi terbaru."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_stats() -> Dict[str, Any]:
    """Statistik ringkas: total prediksi dan jumlah per kelas."""
    conn = get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        per_class_rows = conn.execute(
            "SELECT predicted_class, COUNT(*) AS c FROM predictions GROUP BY predicted_class"
        ).fetchall()
        per_class = {r["predicted_class"]: r["c"] for r in per_class_rows}
        return {"total": total, "per_class": per_class}
    finally:
        conn.close()
