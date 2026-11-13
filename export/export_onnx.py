# Ekspor model Swin (.pth) -> ONNX, verifikasi terhadap PyTorch, lalu unggah ke Hugging Face.
# Dijalankan di Google Colab (sel dipisah oleh "# %%"). Notebook: export_onnx.ipynb

# %% [Sel 1] Pasang library (timm dipin sama dengan versi yang cocok dengan checkpoint)
!pip install -q timm==0.9.16 onnx onnxruntime gdown huggingface_hub

# %% [Sel 2] Unduh checkpoint dari Google Drive
import gdown
DRIVE_ID = "1N5FdfN1W1jfnq1UplPbpyPdb19HeeLlG"
gdown.download(id=DRIVE_ID, output="model_best.pth", quiet=False)

# %% [Sel 3] Muat model (Swin-Tiny, window 3, 3 kelas) dengan strict=True
import torch, timm

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False, num_classes=3, img_size=224, window_size=3,
)
ckpt = torch.load("model_best.pth", map_location="cpu")
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)   # error jika arsitektur tidak cocok
model.eval()
print("Bobot berhasil dimuat. val_acc tersimpan:", ckpt.get("val_acc"))

# %% [Sel 4] Ekspor ke ONNX (batch 1, input 224x224)
dummy = torch.randn(1, 3, 224, 224)
kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=17)
try:
    torch.onnx.export(model, dummy, "model.onnx", dynamo=False, **kwargs)
except TypeError:   # torch lama tanpa argumen dynamo
    torch.onnx.export(model, dummy, "model.onnx", **kwargs)

import os
print("Ukuran model.onnx: %.1f MB" % (os.path.getsize("model.onnx") / 1e6))

# %% [Sel 5] Verifikasi: keluaran ONNX harus sama dengan PyTorch
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
worst = 0.0
for i in range(5):
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    worst = max(worst, float(np.abs(ref - out).max()))
    assert ref.argmax() == out.argmax(), "Prediksi ONNX berbeda dari PyTorch!"
print("Selisih logit maksimum:", worst)
assert worst < 1e-3, "Selisih terlalu besar, jangan diunggah. Kirim pesan ini ke pengembang."
print("VERIFIKASI LOLOS")

# %% [Sel 6] Login Hugging Face (token dengan izin WRITE) lalu unggah
from huggingface_hub import notebook_login, HfApi
notebook_login()

HF_REPO = "bagirshahab/lungvision-swin-onnx"
api = HfApi()
api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="model.onnx", path_in_repo="model.onnx", repo_id=HF_REPO)
print("Selesai: https://huggingface.co/%s/resolve/main/model.onnx" % HF_REPO)
