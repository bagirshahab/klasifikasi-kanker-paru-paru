"""
Model loader & inference untuk klasifikasi kanker paru (Normal / LUAD / LUSC).

Arsitektur: Swin Transformer (mekanisme Shifted Window) dengan transfer learning.
Konfigurasi terbaik dari penelitian:
    - window size : 3x3
    - learning rate: 1e-5
    - weight decay : 0.05
    - akurasi test : 94.96%

Catatan penting soal "window size 3x3":
Varian Swin standar di `timm` (mis. swin_tiny_patch4_window7_224) memakai window 7.
Jika model Anda dibuat dengan window size kustom 3, pastikan nama arsitektur /
konfigurasi di sini SAMA PERSIS dengan yang dipakai saat training, agar bobot .pth
bisa dimuat tanpa error. Sesuaikan MODEL_NAME / img_size bila perlu.
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from torchvision import transforms

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

# Urutan kelas HARUS sama dengan saat training:
# 0 = Normal, 1 = LUAD, 2 = LUSC
CLASS_NAMES: List[str] = ["Normal", "LUAD", "LUSC"]
NUM_CLASSES = len(CLASS_NAMES)

# Ukuran input citra (sesuai training Swin: 224x224)
IMG_SIZE = 224

# Nama arsitektur timm. Ganti sesuai model Anda bila berbeda.
# Default kami pakai swin tiny; jika Anda training dengan window kustom,
# ganti dengan nama/konfigurasi arsitektur yang persis.
MODEL_NAME = os.getenv("MODEL_NAME", "swin_tiny_patch4_window7_224")

# Lokasi file bobot model (.pth). Letakkan file Anda di backend/weights/model_best.pth
WEIGHTS_PATH = os.getenv(
    "WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "weights", "model_best.pth"),
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalisasi ImageNet (standar transfer learning Swin)
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Transformasi inferensi (TANPA augmentasi — hanya resize + normalize)
INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)


# ---------------------------------------------------------------------------
# Pembuatan arsitektur
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """Bangun arsitektur Swin Transformer dengan head 3 kelas."""
    kwargs = {}
    # Window size kustom (mis. WINDOW_SIZE=3). Kosongkan untuk bawaan arsitektur.
    # Default 3 = konfigurasi terbaik penelitian (terverifikasi dari checkpoint:
    # relative_position_bias_table berukuran (2*3-1)^2 = 25).
    window_size = os.getenv("WINDOW_SIZE", "3")
    if window_size:
        kwargs["window_size"] = int(window_size)
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        **kwargs,
    )
    return model


# ---------------------------------------------------------------------------
# Loading bobot — robust terhadap beberapa format checkpoint
# ---------------------------------------------------------------------------

def _extract_state_dict(ckpt):
    """Ambil state_dict dari berbagai kemungkinan format checkpoint."""
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Mungkin dict-nya memang state_dict langsung
        return ckpt

    # Jika checkpoint adalah objek model yang di-torch.save utuh
    try:
        return ckpt.state_dict()
    except AttributeError:
        raise ValueError("Format checkpoint tidak dikenali.")


def _clean_keys(state_dict: dict) -> dict:
    """Hapus prefix umum seperti 'module.' (DataParallel) atau 'model.'."""
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


class LungClassifier:
    """Wrapper untuk load model sekali, lalu dipakai berulang untuk prediksi."""

    def __init__(self, weights_path: str = WEIGHTS_PATH):
        self.weights_path = weights_path
        self.device = DEVICE
        self.model: nn.Module | None = None
        self.loaded = False
        self.load_error: str | None = None
        self._try_load()

    def _download_weights(self) -> None:
        """Unduh bobot dari MODEL_URL jika file belum ada (untuk hosting)."""
        url = os.getenv("MODEL_URL")
        if not url or os.path.exists(self.weights_path):
            return
        import urllib.request

        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        tmp_path = self.weights_path + ".part"
        print(f"[model] Mengunduh bobot dari MODEL_URL ...")
        try:
            import shutil

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            os.replace(tmp_path, self.weights_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[model] Gagal mengunduh bobot: {exc}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _try_load(self) -> None:
        self._download_weights()
        if not os.path.exists(self.weights_path):
            self.load_error = (
                f"File bobot model tidak ditemukan di: {self.weights_path}. "
                "Letakkan file .pth Anda di backend/weights/model_best.pth "
                "(lihat README)."
            )
            return
        try:
            model = build_model()
            ckpt = torch.load(self.weights_path, map_location="cpu")
            state_dict = _clean_keys(_extract_state_dict(ckpt))
            # strict=True: jika arsitektur tidak cocok dengan .pth, gagal dengan jelas
            # (bukan diam-diam menghasilkan prediksi acak).
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model.to(self.device)
            self.model = model
            self.loaded = True
            print(f"[model] Model berhasil dimuat dari {self.weights_path} ke {self.device}")
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"Gagal memuat model: {exc}"
            print(f"[model] ERROR: {self.load_error}")

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        """Prediksi satu citra PIL. Return dict probabilitas + kelas prediksi."""
        if not self.loaded or self.model is None:
            raise RuntimeError(self.load_error or "Model belum dimuat.")

        img = image.convert("RGB")
        tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        pred_idx = int(max(range(NUM_CLASSES), key=lambda i: probs[i]))
        return {
            "predicted_index": pred_idx,
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        }
