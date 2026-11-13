// ---------------------------------------------------------------------------
// LungVision AI — inferensi di browser (onnxruntime-web)
// Model: Swin-Tiny (window 3) hasil ekspor ONNX, input 1x3x224x224.
// ---------------------------------------------------------------------------
const MODEL_URL = window.MODEL_URL;
const CACHE_NAME = "lungvision-model-v1";
const IMG_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406]; // normalisasi ImageNet (sama dengan training)
const STD = [0.229, 0.224, 0.225];
const CLASS_ORDER = ["Normal", "LUAD", "LUSC"]; // urutan label saat training
const HISTORY_KEY = "lungvision_history_v1";
const HISTORY_MAX = 50;

const CLASS_INFO = {
  Normal: {
    label: "Jaringan Normal",
    desc: "Tidak terdeteksi karakteristik keganasan pada citra.",
  },
  LUAD: {
    label: "Lung Adenocarcinoma",
    desc: "Subtipe kanker paru non-sel kecil yang berasal dari sel kelenjar.",
  },
  LUSC: {
    label: "Lung Squamous Cell Carcinoma",
    desc: "Subtipe kanker paru non-sel kecil yang berasal dari sel skuamosa.",
  },
};

const MODEL_META = {
  architecture: "Swin Transformer (Shifted Window)",
  metrics: { accuracy: 0.9496, balanced_accuracy: 0.956, f1_macro: 0.9482 },
  best_config: { window_size: "3x3", learning_rate: "1e-5", weight_decay: 0.05 },
  dataset: "LungHist700 (691 citra H&E, 45 pasien)",
};

// DOM
const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const dropIcon = $("dropIcon");
const dropTitle = $("dropTitle");
const dropHint = $("dropHint");
const fileInput = $("fileInput");

const EMPTY_DEFAULT = $("resultEmpty").innerHTML;
const DROP_TITLE_DEFAULT = dropTitle.innerHTML;
const DROP_HINT_DEFAULT = dropHint.innerHTML;
const predictBtn = $("predictBtn");
const resetBtn = $("resetBtn");
const resultEmpty = $("resultEmpty");
const resultLoading = $("resultLoading");
const resultBox = $("resultBox");
const predClass = $("predClass");
const predSub = $("predSub");
const confidenceVal = $("confidenceVal");
const probsEl = $("probs");
const resultNote = $("resultNote");
const statusDot = $("statusDot");
const statusText = $("statusText");
const loadBar = $("loadBar");
const loadFill = $("loadFill");
const historyBody = $("historyBody");

let selectedFile = null;
let session = null;

// ---------------------------------------------------------------------------
// Model: unduh (dengan cache browser) + buat sesi ONNX
// ---------------------------------------------------------------------------
function setStatus(kind, text) {
  statusDot.className = "status-dot" + (kind ? " " + kind : "");
  statusText.textContent = text;
}

async function fetchModelBytes(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      onProgress(1);
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch {
    cache = null; // Cache API tidak tersedia, lanjut tanpa cache
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Gagal mengunduh model (HTTP ${res.status}).`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress(received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes));
    } catch {
      /* kuota penuh: abaikan, model tetap bisa dipakai */
    }
  }
  return bytes;
}

async function loadModel() {
  setStatus("", "Mengunduh model…");
  loadBar.hidden = false;
  try {
    const bytes = await fetchModelBytes(MODEL_URL, (p) => {
      loadFill.style.width = (p * 100).toFixed(0) + "%";
      setStatus("", `Mengunduh model… ${(p * 100).toFixed(0)}%`);
    });
    setStatus("", "Menyiapkan model…");
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
    });
    setStatus("ok", "Model siap");
    loadBar.hidden = true;
    if (selectedFile) predictBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("err", "Model gagal dimuat");
    loadBar.hidden = true;
    showError("Model gagal dimuat: " + err.message);
  }
}

// ---------------------------------------------------------------------------
// Pra-pemrosesan: resize 224x224, skala 0-1, normalisasi ImageNet, NCHW
// ---------------------------------------------------------------------------
async function toBitmap(file) {
  try {
    return await createImageBitmap(file);
  } catch {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("File bukan citra yang valid."));
      img.src = URL.createObjectURL(file);
    });
  }
}

async function preprocess(file) {
  const bmp = await toBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff"; // latar putih untuk PNG transparan
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, IMG_SIZE, IMG_SIZE);
  const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

  const plane = IMG_SIZE * IMG_SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * plane + i] = (data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return input;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function classify(file) {
  const input = await preprocess(file);
  const tensor = new ort.Tensor("float32", input, [1, 3, IMG_SIZE, IMG_SIZE]);
  const out = await session.run({ input: tensor });
  const probs = softmax(Array.from(out.logits.data));
  const idx = probs.indexOf(Math.max(...probs));
  return {
    predicted_class: CLASS_ORDER[idx],
    confidence: probs[idx],
    probabilities: Object.fromEntries(CLASS_ORDER.map((n, i) => [n, probs[i]])),
  };
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("File harus berupa citra (JPG/PNG).");
    return;
  }
  selectedFile = file;
  dropzone.classList.add("has-file");
  dropIcon.textContent = "\u2713";
  dropTitle.textContent = file.name;
  dropHint.textContent = `${(file.size / 1024).toFixed(0)} KB \u00b7 klik untuk mengganti citra`;
  predictBtn.disabled = !session;
  resetBtn.disabled = false;
}

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  dropzone.classList.remove("has-file");
  dropIcon.textContent = "\u21e7";
  dropTitle.innerHTML = DROP_TITLE_DEFAULT;
  dropHint.innerHTML = DROP_HINT_DEFAULT;
  predictBtn.disabled = true;
  resetBtn.disabled = true;
  resultBox.hidden = true;
  resultLoading.hidden = true;
  resultEmpty.innerHTML = EMPTY_DEFAULT;
  resultEmpty.hidden = false;
});

function showError(message) {
  const p = document.createElement("p");
  p.style.color = "var(--danger)";
  p.textContent = message;
  resultEmpty.replaceChildren(p);
  resultEmpty.hidden = false;
  resultLoading.hidden = true;
  resultBox.hidden = true;
}

// ---------------------------------------------------------------------------
// Prediksi
// ---------------------------------------------------------------------------
predictBtn.addEventListener("click", async () => {
  if (!selectedFile || !session) return;
  resultEmpty.hidden = true;
  resultBox.hidden = true;
  resultLoading.hidden = false;
  predictBtn.disabled = true;
  try {
    // beri browser kesempatan menampilkan spinner sebelum komputasi berat
    await new Promise((r) => setTimeout(r, 30));
    const result = await classify(selectedFile);
    renderResult(result);
    saveHistory(selectedFile.name, result);
    renderHistory();
  } catch (err) {
    console.error(err);
    showError("Gagal melakukan prediksi: " + err.message);
  } finally {
    predictBtn.disabled = false;
  }
});

function renderResult(data) {
  resultLoading.hidden = true;
  resultBox.hidden = false;
  const cls = data.predicted_class;
  const info = CLASS_INFO[cls];

  predClass.textContent = cls;
  predClass.className = `result-class ${cls}`;
  predSub.textContent = info.label;
  confidenceVal.textContent = (data.confidence * 100).toFixed(2) + "%";

  probsEl.replaceChildren();
  CLASS_ORDER.forEach((name) => {
    const p = data.probabilities[name] * 100;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-top"><span>${name}</span><span>${p.toFixed(2)}%</span></div>
      <div class="prob-bar"><div class="prob-fill ${name}" style="width:${p}%"></div></div>`;
    probsEl.appendChild(row);
  });
  resultNote.textContent = info.desc;
}

// ---------------------------------------------------------------------------
// Info model
// ---------------------------------------------------------------------------
function renderModelInfo() {
  const m = MODEL_META;
  $("modelInfo").innerHTML = `
    <div class="info-item"><div class="k">Arsitektur</div><div class="v" style="font-size:0.95rem">${m.architecture}</div></div>
    <div class="info-item"><div class="k">Akurasi</div><div class="v">${(m.metrics.accuracy * 100).toFixed(2)}%</div></div>
    <div class="info-item"><div class="k">Balanced Acc</div><div class="v">${m.metrics.balanced_accuracy.toFixed(4)}</div></div>
    <div class="info-item"><div class="k">F1-Macro</div><div class="v">${m.metrics.f1_macro.toFixed(4)}</div></div>
    <div class="info-item info-full"><div class="k">Konfigurasi terbaik</div>
      <div class="v" style="font-size:0.95rem">Window ${m.best_config.window_size} &middot; LR ${m.best_config.learning_rate} &middot; Weight decay ${m.best_config.weight_decay}</div></div>
    <div class="info-item info-full"><div class="k">Dataset</div>
      <div class="v" style="font-size:0.95rem">${m.dataset}</div></div>`;
}

// ---------------------------------------------------------------------------
// Riwayat (localStorage)
// ---------------------------------------------------------------------------
function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function saveHistory(filename, result) {
  const items = readHistory();
  const nextId = items.length ? items[0].id + 1 : 1;
  items.unshift({
    id: nextId,
    filename,
    predicted_class: result.predicted_class,
    confidence: result.confidence,
    created_at: new Date().toISOString(),
  });
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX)));
  } catch {
    /* penyimpanan penuh/diblokir: abaikan */
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function renderHistory() {
  const items = readHistory();
  if (!items.length) {
    historyBody.innerHTML = `<tr><td colspan="5" class="muted">Belum ada riwayat.</td></tr>`;
    return;
  }
  historyBody.innerHTML = items
    .map((it) => {
      const cls = CLASS_ORDER.includes(it.predicted_class) ? it.predicted_class : "";
      return `<tr>
        <td>${Number(it.id)}</td>
        <td>${escapeHtml(it.filename)}</td>
        <td><span class="badge ${cls}">${escapeHtml(it.predicted_class)}</span></td>
        <td>${(Number(it.confidence) * 100).toFixed(2)}%</td>
        <td>${escapeHtml(new Date(it.created_at).toLocaleString("id-ID"))}</td>
      </tr>`;
    })
    .join("");
}

$("clearHistory").addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

// Init
renderModelInfo();
renderHistory();
loadModel();
