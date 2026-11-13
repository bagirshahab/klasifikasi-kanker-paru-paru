// ---------------------------------------------------------------------------
// Konfigurasi API
// Jika frontend disajikan oleh backend yang sama, biarkan "" (relatif).
// Jika frontend & backend terpisah, isi dengan URL backend, mis:
//   const API_BASE = "http://localhost:8000";
// ---------------------------------------------------------------------------
const API_BASE = "";

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

const CLASS_ORDER = ["Normal", "LUAD", "LUSC"];

// Elemen DOM
const dropzone = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzoneInner");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");

const resultEmpty = document.getElementById("resultEmpty");
const resultLoading = document.getElementById("resultLoading");
const resultBox = document.getElementById("resultBox");
const predClass = document.getElementById("predClass");
const predSub = document.getElementById("predSub");
const confidenceVal = document.getElementById("confidenceVal");
const probsEl = document.getElementById("probs");
const resultNote = document.getElementById("resultNote");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const modelInfoEl = document.getElementById("modelInfo");
const historyBody = document.getElementById("historyBody");
const refreshHistory = document.getElementById("refreshHistory");

let selectedFile = null;

// ---------------------------------------------------------------------------
// Upload handling
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
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropzoneInner.hidden = true;
  };
  reader.readAsDataURL(file);
  predictBtn.disabled = false;
  resetBtn.disabled = false;
}

resetBtn.addEventListener("click", resetAll);

function resetAll() {
  selectedFile = null;
  fileInput.value = "";
  preview.src = "";
  preview.hidden = true;
  dropzoneInner.hidden = false;
  predictBtn.disabled = true;
  resetBtn.disabled = true;
  resultBox.hidden = true;
  resultLoading.hidden = true;
  resultEmpty.hidden = false;
}

// ---------------------------------------------------------------------------
// Prediksi
// ---------------------------------------------------------------------------
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  resultEmpty.hidden = true;
  resultBox.hidden = true;
  resultLoading.hidden = false;
  predictBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Gagal melakukan prediksi.");
    renderResult(data);
    loadHistory();
  } catch (err) {
    resultLoading.hidden = true;
    resultEmpty.hidden = false;
    const p = document.createElement("p");
    p.style.color = "var(--danger)";
    p.textContent = err.message;
    resultEmpty.replaceChildren(p);
  } finally {
    predictBtn.disabled = false;
  }
});

function renderResult(data) {
  resultLoading.hidden = true;
  resultBox.hidden = false;

  const cls = data.predicted_class;
  const info = CLASS_INFO[cls] || { label: cls, desc: "" };

  predClass.textContent = cls;
  predClass.className = `result-class ${cls}`;
  predSub.textContent = info.label;
  confidenceVal.textContent = (data.confidence * 100).toFixed(2) + "%";

  // Bar probabilitas (urut tetap Normal, LUAD, LUSC)
  probsEl.innerHTML = "";
  CLASS_ORDER.forEach((name) => {
    const p = (data.probabilities[name] || 0) * 100;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-top">
        <span>${name}</span>
        <span>${p.toFixed(2)}%</span>
      </div>
      <div class="prob-bar">
        <div class="prob-fill ${name}" style="width:${p}%"></div>
      </div>`;
    probsEl.appendChild(row);
  });

  resultNote.textContent = info.desc;
}

// ---------------------------------------------------------------------------
// Status model + info
// ---------------------------------------------------------------------------
async function loadHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.model_loaded) {
      statusDot.className = "status-dot ok";
      statusText.textContent = "Model siap";
    } else {
      statusDot.className = "status-dot err";
      statusText.textContent = "Model belum dimuat";
      predictBtn.title = data.model_error || "";
    }
  } catch {
    statusDot.className = "status-dot err";
    statusText.textContent = "Backend tidak terhubung";
  }
}

async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/api/model-info`);
    const m = await res.json();
    const acc = (m.metrics.accuracy * 100).toFixed(2);
    const ba = (m.metrics.balanced_accuracy).toFixed(4);
    const f1 = (m.metrics.f1_macro).toFixed(4);
    modelInfoEl.innerHTML = `
      <div class="info-item"><div class="k">Arsitektur</div><div class="v" style="font-size:0.95rem">${m.architecture}</div></div>
      <div class="info-item"><div class="k">Akurasi</div><div class="v">${acc}%</div></div>
      <div class="info-item"><div class="k">Balanced Acc</div><div class="v">${ba}</div></div>
      <div class="info-item"><div class="k">F1-Macro</div><div class="v">${f1}</div></div>
      <div class="info-item info-full">
        <div class="k">Konfigurasi terbaik</div>
        <div class="v" style="font-size:0.95rem">
          Window ${m.best_config.window_size} &middot; LR ${m.best_config.learning_rate} &middot; Weight decay ${m.best_config.weight_decay}
        </div>
      </div>
      <div class="info-item info-full">
        <div class="k">Dataset</div>
        <div class="v" style="font-size:0.95rem">${m.dataset}</div>
      </div>`;
  } catch {
    modelInfoEl.innerHTML = `<p class="muted">Gagal memuat info model.</p>`;
  }
}

// ---------------------------------------------------------------------------
// Riwayat
// ---------------------------------------------------------------------------
async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/history?limit=20`);
    const data = await res.json();
    if (!data.items.length) {
      historyBody.innerHTML = `<tr><td colspan="5" class="muted">Belum ada riwayat.</td></tr>`;
      return;
    }
    historyBody.innerHTML = data.items
      .map((it) => {
        const t = new Date(it.created_at);
        const timeStr = isNaN(t) ? it.created_at : t.toLocaleString("id-ID");
        return `<tr>
          <td>${it.id}</td>
          <td>${escapeHtml(it.filename)}</td>
          <td><span class="badge ${it.predicted_class}">${it.predicted_class}</span></td>
          <td>${(it.confidence * 100).toFixed(2)}%</td>
          <td>${timeStr}</td>
        </tr>`;
      })
      .join("");
  } catch {
    historyBody.innerHTML = `<tr><td colspan="5" class="muted">Gagal memuat riwayat.</td></tr>`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

refreshHistory.addEventListener("click", loadHistory);

// Init
loadHealth();
loadModelInfo();
loadHistory();
