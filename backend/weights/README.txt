Letakkan file bobot model terbaik Anda di folder ini dengan nama:

    model_best.pth

Jadi path lengkapnya:  backend/weights/model_best.pth

Model yang diharapkan: Swin Transformer (konfigurasi terbaik window 3x3,
learning rate 1e-5, weight decay 0.05) dengan 3 kelas output:
    0 = Normal, 1 = LUAD, 2 = LUSC

Jika nama file atau arsitektur berbeda, sesuaikan variabel WEIGHTS_PATH /
MODEL_NAME (lihat backend/model.py atau pakai environment variable).
