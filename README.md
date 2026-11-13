# LungVision AI — Klasifikasi Kanker Paru (Swin Transformer)

Website untuk implementasi hasil training model **Swin Transformer** klasifikasi
subtipe kanker paru pada citra histopatologi H&E:

- **Normal** — jaringan normal
- **LUAD** — Lung Adenocarcinoma
- **LUSC** — Lung Squamous Cell Carcinoma

Model terbaik dari penelitian (window size 3×3, learning rate 1×10⁻⁵, weight
decay 0,05): **akurasi 94,96%**, balanced accuracy 0,9560, F1-Macro 0,9482.
Dataset: LungHist700 (691 citra H&E, 45 pasien).

---

## 1. Struktur Proyek

```
lungvision-ai/
├── backend/                 # API Python (FastAPI) + inferensi model
│   ├── main.py              # endpoint API + penyajian frontend
│   ├── model.py             # loader Swin Transformer + preprocessing + predict
│   ├── database.py          # SQLite riwayat prediksi
│   ├── requirements.txt     # dependency Python
│   ├── weights/             # >>> LETAKKAN model_best.pth DI SINI <<<
│   │   └── README.txt
│   └── data/                # database SQLite (dibuat otomatis saat runtime)
├── frontend/                # tampilan web
│   ├── index.html
│   ├── style.css
│   └── script.js
├── .gitignore
└── README.md
```

---

## 2. Di Mana Menyimpan Hasil Training (file .pth)?

**Letakkan file bobot model Anda di:**

```
backend/weights/model_best.pth
```

Beberapa catatan penting:

1. **Nama file.** Default yang dicari aplikasi adalah `model_best.pth`. Kalau nama
   file Anda berbeda, ada dua pilihan:
   - ganti namanya menjadi `model_best.pth`, atau
   - set environment variable `WEIGHTS_PATH` ke path file Anda.

2. **Arsitektur harus cocok.** `backend/model.py` membangun ulang arsitektur Swin
   lalu memuat bobot ke dalamnya. Pastikan nama arsitektur `MODEL_NAME` SAMA dengan
   yang Anda pakai saat training. Default: `swin_tiny_patch4_window7_224`.
   - Kalau model Anda dibuat dengan **window size kustom 3** (bukan 7 bawaan Swin
     standar), ganti `MODEL_NAME` / konfigurasi di `model.py` agar persis sama
     dengan definisi arsitektur saat training. Jika tidak cocok, bobot gagal dimuat.
   - Window size bisa diatur lewat environment variable, mis.
     `$env:WINDOW_SIZE="3"` sebelum menjalankan uvicorn.
   - Bobot dimuat dengan `strict=True`: kalau arsitektur tidak cocok, server
     menampilkan error jelas di `/api/health` (bukan prediksi acak).
   - Loader sudah dibuat robust: menerima checkpoint berupa `state_dict`,
     `{"state_dict": ...}`, `{"model_state_dict": ...}`, atau objek model utuh,
     dan otomatis membersihkan prefix `module.` / `model.`.

3. **Jangan commit file .pth ke Git** (ukurannya besar). File `.gitignore` sudah
   mengabaikan `backend/weights/*.pth`. Lihat bagian **Opsi penyimpanan model** di
   bawah untuk distribusi model.

### Cara mengirim file .pth ke saya / ke repository

Karena ekstensi `.pth` tidak bisa dilampirkan di chat, gunakan salah satu cara:

- **Google Drive / cloud** — unggah `.pth`, bagikan tautannya. Nanti tinggal
  diunduh ke `backend/weights/`.
- **GitHub Release / Git LFS** — simpan model sebagai aset rilis atau pakai
  [Git LFS](https://git-lfs.com/) untuk file besar (lihat bagian Deployment).
- **Hugging Face Hub** — unggah model, lalu unduh via URL.

---

## 3. Menjalankan Secara Lokal

### Prasyarat
- **Python 3.10 atau 3.11** (versi di `requirements.txt` belum punya paket
  PyTorch untuk Python 3.13/3.14). Di Windows: `py -3.11 -m venv .venv`
- (opsional) GPU + CUDA untuk inferensi lebih cepat; tanpa GPU tetap jalan di CPU.

### Langkah

```powershell
# 1. Masuk ke folder backend
cd lungvision-ai/backend

# 2. Buat virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# 3. Install dependency
pip install -r requirements.txt

# 4. Pastikan model ada di backend/weights/model_best.pth

# 5. Jalankan server
uvicorn main:app --reload --port 8000
```

Buka browser ke **http://localhost:8000** — frontend akan otomatis disajikan oleh
backend yang sama (tidak perlu server terpisah untuk frontend).

> Catatan: dependency `torch` cukup besar (ratusan MB). Instalasi pertama bisa lama.
> Jika ingin versi CPU-only yang lebih ringan, ikuti petunjuk di
> https://pytorch.org/get-started/locally/

---

## 4. Database — Langkah & Penjelasan

Aplikasi memakai **SQLite**, database berbasis file yang **tidak butuh server
terpisah**. Cocok untuk mencatat riwayat prediksi website.

- **Kapan dibuat?** Otomatis saat server pertama kali dijalankan. File dibuat di
  `backend/data/predictions.db`. Anda tidak perlu meng-install apa pun.
- **Tabel yang dibuat:** `predictions`
  | kolom | arti |
  |-------|------|
  | `id` | nomor urut |
  | `filename` | nama file citra yang diunggah |
  | `predicted_class` | hasil prediksi (Normal/LUAD/LUSC) |
  | `confidence` | probabilitas kelas pemenang |
  | `prob_normal`, `prob_luad`, `prob_lusc` | probabilitas tiap kelas |
  | `created_at` | waktu prediksi (UTC) |
- **Dipakai oleh endpoint:** `/api/history` (riwayat) dan `/api/stats` (statistik).
- **Reset database:** cukup hapus file `backend/data/predictions.db`, tabel akan
  dibuat ulang saat server dijalankan lagi.

### Mau pakai PostgreSQL (mis. Neon) sebagai gantinya?
Ganti isi `backend/database.py` agar koneksi mengarah ke Postgres (library
`psycopg` atau `@neondatabase/serverless` jika lewat Node). Skema tabel di atas
bisa dipakai langsung dengan sedikit penyesuaian tipe (`SERIAL`, `TIMESTAMPTZ`,
`DOUBLE PRECISION`). SQLite sudah cukup untuk kebutuhan demo/skripsi.

---

## 5. Endpoint API

| Method | Path | Fungsi |
|--------|------|--------|
| GET | `/api/health` | status server & apakah model sudah dimuat |
| GET | `/api/model-info` | metadata arsitektur + metrik penelitian |
| POST | `/api/predict` | upload citra (`multipart/form-data`, field `file`), kembalikan prediksi |
| GET | `/api/history?limit=20` | riwayat prediksi terbaru |
| GET | `/api/stats` | total prediksi & jumlah per kelas |

Contoh respons `/api/predict`:

```json
{
  "predicted_index": 1,
  "predicted_class": "LUAD",
  "confidence": 0.904,
  "probabilities": { "Normal": 0.05, "LUAD": 0.904, "LUSC": 0.046 },
  "record_id": 12,
  "filename": "aca_md_40x_92.jpg"
}
```

---

## 6. Deployment (opsional)

Karena inferensi butuh PyTorch, deployment berbeda dari situs statis biasa:

- **Hugging Face Spaces (rekomendasi, gratis)** — jenis "Docker" atau "Gradio/
  FastAPI". Cocok untuk model PyTorch. Unggah seluruh folder + model `.pth`.
- **Railway / Render / Fly.io** — mendukung container Python; set perintah start
  `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **VPS sendiri** — jalankan dengan `uvicorn` + reverse proxy (nginx).

Untuk model besar di Git, gunakan **Git LFS**:

```powershell
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add backend/weights/model_best.pth
git commit -m "Add model weights via LFS"
```

---

## 7. Disclaimer

Alat ini untuk keperluan **riset dan edukasi**. Hasil klasifikasi **bukan
diagnosis medis** dan tidak menggantikan penilaian ahli patologi.

---

## 8. Deploy: Frontend di Vercel + Backend di Hugging Face Spaces

Vercel tidak bisa menjalankan backend ini (PyTorch melebihi batas ukuran fungsi
dan SQLite tidak bisa ditulis). Karena itu hosting dipisah:

**A. Backend → Hugging Face Spaces**
1. Buat Space baru di huggingface.co, pilih SDK **Docker**.
2. Di repo Space, salin isi repo ini (`Dockerfile`, `backend/`, `frontend/`) dan
   tambahkan front matter ini di paling atas `README.md` milik Space:
   ```
   ---
   title: LungVision AI
   sdk: docker
   app_port: 7860
   ---
   ```
3. Model: **tidak perlu diunggah manual.** `Dockerfile` sudah berisi `MODEL_URL`
   (Google Drive) dan bobot diunduh otomatis saat server pertama jalan. Untuk
   memakai model lain, timpa Variable `MODEL_URL` di Settings Space.
4. Arsitektur default sudah sesuai checkpoint (Swin-Tiny, window 3). Variable
   `MODEL_NAME` / `WINDOW_SIZE` hanya perlu diisi jika modelnya diganti.
5. Tunggu build selesai, cek `https://USERNAME-NAMA-SPACE.hf.space/api/health`.

**B. Frontend → Vercel**
1. Edit `frontend/config.js`, isi `window.API_BASE` dengan URL Space di atas.
2. Commit & push. Vercel membaca `vercel.json` dan menyajikan folder `frontend/`.

Catatan: database SQLite di Space bersifat sementara (hilang saat restart).
Untuk riwayat permanen gunakan database eksternal (mis. Neon).
