# LungVision AI — Klasifikasi Kanker Paru (Swin Transformer)

Website klasifikasi subtipe kanker paru pada citra histopatologi H&E:
**Normal**, **LUAD** (Adenocarcinoma), **LUSC** (Squamous Cell Carcinoma).

Model: Swin-Tiny, window 3×3, lr 1e-5, weight decay 0,05 — akurasi 94,96%,
balanced accuracy 0,9560, F1-Macro 0,9482 (dataset LungHist700).

## Cara kerja

Tidak ada server. Model diekspor ke ONNX dan **dijalankan langsung di browser**
pengunjung dengan `onnxruntime-web`. Citra tidak pernah diunggah ke mana pun.

```
Vercel (situs statis: frontend/)  ──fetch──►  Hugging Face (repo model: model.onnx)
```

- Frontend: folder `frontend/` (disajikan Vercel lewat `vercel.json`).
- Model: `model.onnx` (~110 MB) di repo model Hugging Face, diunduh sekali lalu
  disimpan di cache browser.
- Riwayat prediksi: `localStorage` browser (per perangkat).

## Langkah deploy

### 1. Ekspor model ke ONNX (sekali saja, di Google Colab)
1. Buka https://colab.research.google.com → Upload → pilih `export/export_onnx.ipynb`.
2. Jalankan sel 1 sampai 5 berurutan. Sel 5 harus mencetak `VERIFIKASI LOLOS`
   (membuktikan keluaran ONNX sama dengan PyTorch).
3. Buat token di https://huggingface.co/settings/tokens (tipe **Write**), lalu
   jalankan sel 6 dan tempel token. Model diunggah ke
   `bagirshahab/lungvision-swin-onnx`.

### 2. Frontend
`frontend/config.js` sudah menunjuk ke
`https://huggingface.co/bagirshahab/lungvision-swin-onnx/resolve/main/model.onnx`.
Cukup push ke GitHub, Vercel deploy otomatis.

## Menjalankan lokal
```powershell
cd frontend
python -m http.server 8000     # lalu buka http://localhost:8000
```

## Catatan teknis
- Pra-pemrosesan di browser: resize 224×224, skala 0–1, normalisasi ImageNet
  (mean 0.485/0.456/0.406, std 0.229/0.224/0.225). Harus sama dengan training.
  Resize memakai canvas browser, sedikit berbeda dari PIL/torchvision, sehingga
  probabilitas bisa selisih tipis dari hasil notebook.
- Urutan kelas: 0 = Normal, 1 = LUAD, 2 = LUSC.

## Disclaimer
Alat ini untuk **riset dan edukasi**. Hasil klasifikasi **bukan diagnosis medis**
dan tidak menggantikan penilaian ahli patologi.
