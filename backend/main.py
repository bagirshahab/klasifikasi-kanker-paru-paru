"""
API FastAPI untuk website klasifikasi kanker paru (Normal / LUAD / LUSC).

Menjalankan:
    uvicorn main:app --reload --port 8000

Endpoint:
    GET  /api/health      -> status server & model
    GET  /api/model-info  -> metadata model & metrik dari penelitian
    POST /api/predict     -> upload citra, kembalikan hasil klasifikasi
    GET  /api/history     -> riwayat prediksi terbaru
    GET  /api/stats       -> statistik ringkas
Frontend statis disajikan di root "/".
"""

from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import database
from model import LungClassifier, CLASS_NAMES, MODEL_NAME

app = FastAPI(title="LungVision AI — Klasifikasi Kanker Paru", version="1.0.0")

# CORS: izinkan frontend diakses dari mana saja (dev). Batasi di produksi.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metadata model dari hasil penelitian (konfigurasi terbaik)
MODEL_META = {
    "architecture": "Swin Transformer (Shifted Window)",
    "model_name": MODEL_NAME,
    "classes": CLASS_NAMES,
    "best_config": {"window_size": "3x3", "learning_rate": "1e-5", "weight_decay": 0.05},
    "metrics": {
        "accuracy": 0.9496,
        "balanced_accuracy": 0.9560,
        "f1_macro": 0.9482,
        "f1_weighted": 0.9498,
        "precision_macro": 0.9444,
        "recall_macro": 0.9560,
    },
    "per_class": {
        "Normal": {"precision": 0.886, "recall": 1.000, "f1": 0.939},
        "LUAD": {"precision": 0.947, "recall": 0.964, "f1": 0.956},
        "LUSC": {"precision": 1.000, "recall": 0.904, "f1": 0.949},
    },
    "dataset": "LungHist700 (691 citra H&E, 45 pasien)",
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Inisialisasi model & database saat startup
classifier = LungClassifier()


@app.on_event("startup")
def _startup() -> None:
    database.init_db()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier.loaded,
        "model_error": classifier.load_error,
    }


@app.get("/api/model-info")
def model_info():
    return {**MODEL_META, "model_loaded": classifier.loaded}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not classifier.loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum siap. " + (classifier.load_error or "")
            ),
        )

    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa citra (image).")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="File kosong.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran file maksimal 10 MB.")

    try:
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File bukan citra yang valid.")

    try:
        result = classifier.predict(image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Gagal melakukan inferensi: {exc}")

    # Simpan ke riwayat
    try:
        record_id = database.save_prediction(file.filename or "unknown", result)
        result["record_id"] = record_id
    except Exception as exc:  # noqa: BLE001
        # Jangan gagalkan prediksi hanya karena gagal simpan riwayat
        print(f"[db] gagal simpan riwayat: {exc}")

    result["filename"] = file.filename
    return result


@app.get("/api/history")
def history(limit: int = 20):
    limit = max(1, min(limit, 100))
    return {"items": database.get_history(limit)}


@app.get("/api/stats")
def stats():
    return database.get_stats()


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(FRONTEND_DIR):

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
