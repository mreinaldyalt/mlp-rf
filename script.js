window.addEventListener("error", (e) => {
  alert("JS ERROR: " + (e?.message || e));
});
window.addEventListener("unhandledrejection", (e) => {
  alert("PROMISE ERROR: " + (e?.reason?.message || e?.reason || e));
});


const uploadButton = document.getElementById("uploadButton");
const fileInput = document.getElementById("fileInput");
const fileNameText = document.getElementById("fileName");
const previewSection = document.getElementById("previewSection");
const previewTable = document.getElementById("previewTable");

// Tombol algoritma
const btnMLP = document.getElementById("btnMLP");
const btnRF = document.getElementById("btnRF");
const btnSummary = document.getElementById("btnSummary");

// Elemen modal
const modalOverlay = document.getElementById("modalOverlay");
const modalTitle = document.getElementById("modalTitle");
const modalBody = document.getElementById("modalBody");
const modalClose = document.getElementById("modalClose");
const modalOk = document.getElementById("modalOk");

// Section ringkasan gabungan
const summarySection = document.getElementById("summarySection");
const summaryContent = document.getElementById("summaryContent");

// Menyimpan teks CSV mentah + file asli
let csvRawText = "";
let currentFile = null;

// Menyimpan hasil terakhir MLP & RF (untuk kesimpulan gabungan & reuse popup)
let lastMlpResult = null;
let lastRfResult = null;

// ✅ FIX: fungsi ini WAJIB ada karena dipanggil saat upload & setelah training
function updateSummaryButtonState() {
  if (!btnSummary) return;

  if (lastMlpResult && lastRfResult) {
    btnSummary.disabled = false;
    btnSummary.classList.add("enabled");
  } else {
    btnSummary.disabled = true;
    btnSummary.classList.remove("enabled");
  }
}


/* ======================================================
   Upload & preview CSV
====================================================== */

uploadButton.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  currentFile = file; // simpan file asli untuk backend

  // Reset hasil MLP & RF setiap kali ganti file
  lastMlpResult = null;
  lastRfResult = null;
  updateSummaryButtonState();
  if (summarySection) {
    summarySection.classList.add("hidden");
    summaryContent.innerHTML = "";
  }

  // Validasi tipe file
  if (file.type !== "text/csv" && !file.name.toLowerCase().endsWith(".csv")) {
    alert("Tolong upload file dengan ekstensi .csv");
    fileInput.value = "";
    return;
  }
  fileNameText.textContent = `File terpilih: ${file.name}`;
fileNameText.dataset.baseName = fileNameText.textContent;
setUploadStatus("📌 Menunggu proses...");

  const reader = new FileReader();
  reader.onload = async (event) => {
  const text = event.target.result;
  csvRawText = text;

  // tampilkan preview dulu (fungsi lama jangan hilang)
  renderCsvPreview(text);

  // notifikasi langsung (biar user tau lagi proses)
  showModal("Data Preparation", `
    <p><strong>Loading...</strong></p>
    <p>Menjalankan data preparation di backend (Render kadang butuh beberapa detik kalau baru bangun).</p>
  `);
  setUploadStatus("⏳ Menjalankan data preparation...");

  try {
    const formData = new FormData();
    formData.append("file", currentFile);

    const res = await fetchWithTimeout(
      PREPARE_API_URL,
      { method: "POST", body: formData },
      60000
    );

    // kalau CORS / 502, kasih error yang kebaca
    if (!res.ok) {
      const t = await res.text();
      throw new Error(`Prepare gagal. Status ${res.status}: ${t}`);
    }

    const data = await res.json();

    if (data.error) {
      showModal("Data Preparation Error", `<p>${data.error}</p>`);
      setUploadStatus("❌ Data preparation gagal (lihat modal).");
      return;
    }

    // tampilkan ke modal (yang sudah ada)
    renderDataPreparation(data);

    // tampilkan juga ke halaman (processSection) biar sesuai request kamu
    renderPreparationToProcessSection(data);

    setUploadStatus("✅ Data preparation selesai. Silakan pilih MLP / RF.");

  } catch (err) {
    showModal(
      "Data Preparation Error",
      `<p>Gagal memproses data preparation:<br><code>${err.message}</code></p>
       <p>Catatan: kalau Render baru bangun (cold start), coba upload lagi 1x.</p>`
    );
    setUploadStatus("❌ Data preparation gagal (cek modal).");
  }
};



  reader.onerror = () => {
    alert("Gagal membaca file CSV.");
  };

  reader.readAsText(file);
});

// deteksi delimiter , atau ;
function detectDelimiter(line) {
  const commaCount = (line.match(/,/g) || []).length;
  const semicolonCount = (line.match(/;/g) || []).length;
  return semicolonCount > commaCount ? ";" : ",";
}

// render preview CSV
function renderCsvPreview(csvText) {
  const rows = csvText.trim().split(/\r?\n/);
  if (rows.length === 0) return;

  const delimiter = detectDelimiter(rows[0]);
  const maxRows = 50;

  previewTable.innerHTML = "";

  rows.slice(0, maxRows).forEach((row, rowIndex) => {
    if (!row.trim()) return;

    const tr = document.createElement("tr");
    const columns = row.split(delimiter);

    columns.forEach((col) => {
      const cell = document.createElement(rowIndex === 0 ? "th" : "td");
      cell.textContent = col;
      tr.appendChild(cell);
    });

    previewTable.appendChild(tr);
  });

  previewSection.classList.remove("hidden");
}
function renderDataPreparation(data) {
  const stepsHtml = data.steps
    .map(
      (s) =>
        `<li><strong>${s.title}</strong><br><span>${s.detail}</span></li>`
    )
    .join("");

  const previewRows = data.preview
    .map(
      (row) =>
        `<tr>${Object.values(row)
          .map((v) => `<td>${v}</td>`)
          .join("")}</tr>`
    )
    .join("");

  const previewHead = data.columns
    .map((c) => `<th>${c}</th>`)
    .join("");

  const html = `
    <h3>Hasil Data Preparation</h3>
    <p><strong>Ukuran data bersih:</strong> ${data.clean_shape[0]} baris × ${data.clean_shape[1]} kolom</p>

    <h4>Langkah-langkah</h4>
    <ol>${stepsHtml}</ol>

    <h4>Preview Data Bersih</h4>
    <div class="table-wrapper">
      <table>
        <tr>${previewHead}</tr>
        ${previewRows}
      </table>
    </div>

    <p style="margin-top:12px;">
      ✅ Data sudah siap. Silakan pilih <strong>Algoritma MLP</strong> atau <strong>Random Forest</strong>.
    </p>
  `;

  showModal("Data Preparation Selesai", html);
}
function renderPreparationToProcessSection(data) {
  // pakai section proses yang sudah ada
  const processSection = document.getElementById("processSection");
  const processMeta = document.getElementById("processMeta");
  const processStepsEl = document.getElementById("processSteps");

  if (!processSection || !processMeta || !processStepsEl) return;

  processSection.classList.remove("hidden");

  processMeta.innerHTML = `
    <div style="padding:10px;border-radius:10px;border:1px solid rgba(255,255,255,.15);">
      <strong>Tahap:</strong> Data Preparation (otomatis setelah upload)<br/>
      <strong>Ukuran data bersih:</strong> ${data.clean_shape[0]} baris × ${data.clean_shape[1]} kolom
    </div>
  `;

  processStepsEl.innerHTML = "";
  (data.steps || []).forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${s.title}</strong><br/><span style="opacity:.9">${s.detail}</span>`;
    processStepsEl.appendChild(li);
  });

  // scroll ke section proses biar user langsung lihat
  processSection.scrollIntoView({ behavior: "smooth" });
}

// =======================
// BASE URL UNTUK BACKEND (HARUS DI ATAS SEBELUM DIPAKAI)
// =======================
const BACKEND_BASE_URL = "https://mlp-rf.onrender.com";

const PREPARE_API_URL = `${BACKEND_BASE_URL}/prepare`;
const MLP_API_URL = `${BACKEND_BASE_URL}/mlp`;
const RF_API_URL  = `${BACKEND_BASE_URL}/rf`;

// helper: fetch dengan timeout (biar kalau Render lagi bangun, tetap ada feedback)
async function fetchWithTimeout(url, options = {}, timeoutMs = 60000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

// helper: status kecil di bawah nama file
function setUploadStatus(msg) {
  // aman: status nempel di bawah fileNameText
  const base = fileNameText.dataset.baseName || fileNameText.textContent || "";
  if (!fileNameText.dataset.baseName) fileNameText.dataset.baseName = base;
  fileNameText.textContent = `${fileNameText.dataset.baseName}\n${msg}`;
  fileNameText.style.whiteSpace = "pre-line";
}


async function callMlpApi(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(MLP_API_URL, { method: "POST", body: formData });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Status ${res.status}: ${text}`);
  }

  return await res.json();
}

async function callRfApi(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(RF_API_URL, { method: "POST", body: formData });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Status ${res.status}: ${text}`);
  }

  return await res.json();
}

/* ======================================================
   Pilihan algoritma
====================================================== */

btnMLP.addEventListener("click", () => {
  handleAlgorithmClick("MLP");
});

btnRF.addEventListener("click", () => {
  handleAlgorithmClick("RF");
});

btnSummary.addEventListener("click", () => {
  showGlobalSummary();
});

async function handleAlgorithmClick(algoName) {
  if (!csvRawText || !currentFile) {
    alert("Silakan upload file CSV terlebih dahulu sebelum memilih algoritma.");
    return;
  }

  /* ==========================
     MLP via backend (Python)
     ========================== */
  if (algoName === "MLP") {
    try {
      // Jika sudah pernah dihitung untuk file ini, pakai hasil cache saja
      if (lastMlpResult) {
        renderMlpModal(lastMlpResult);
        return;
      }

      showModal("Hasil Algoritma MLP", "<p>Sedang menjalankan MLP di backend...</p>");

      const data = await callMlpApi(currentFile);

      if (data.error) {
        showModal("Hasil Algoritma MLP", `<p>${data.error}</p>`);
        return;
      }

      lastMlpResult = data;
      updateSummaryButtonState();
      renderMlpModal(data);
    } catch (err) {
      showModal(
        "Hasil Algoritma MLP",
        `<p>Terjadi error saat memanggil backend MLP:<br><code>${err.message}</code></p>`
      );
    }
    return;
  }

  /* ==========================
     Random Forest via backend
     ========================== */
  if (algoName === "RF") {
    try {
      // Jika sudah pernah dihitung untuk file ini, pakai hasil cache saja
      if (lastRfResult) {
        renderRfModal(lastRfResult);
        return;
      }

      showModal(
        "Hasil Algoritma Random Forest",
        "<p>Sedang menjalankan Random Forest di backend...</p>"
      );

      const data = await callRfApi(currentFile);

      if (data.error) {
        showModal("Hasil Algoritma Random Forest", `<p>${data.error}</p>`);
        return;
      }

      lastRfResult = data;
      updateSummaryButtonState();
      renderRfModal(data);
    } catch (err) {
      showModal(
        "Hasil Algoritma Random Forest",
        `<p>Terjadi error saat memanggil backend Random Forest:<br><code>${err.message}</code></p>`
      );
    }
    return;
  }

  // fallback (harusnya sudah tidak kepakai)
  const htmlResult = generateDummyResult(algoName, csvRawText);
  showModal(`Hasil Algoritma ${algoName}`, htmlResult);
}

/* ======================================================
   Render Modal MLP & RF dari data cache
====================================================== */

function renderMlpModal(data) {
  // Tetap pertahankan modal (fungsi lama tetap ada)
  const m = data.metrics;
  const summary = data.summary;
  const conclusion = data.conclusion || "Backend tidak mengirim kesimpulan.";
  const lossImg = data.plots.loss_curve;
  const scatterImg = data.plots.pred_vs_actual;

  const htmlResult = `
    <p>Training selesai menggunakan <strong>Multi-Layer Perceptron (MLP)</strong> di backend Python.</p>
    <ul>
      <li>Jumlah fitur: <strong>${m.n_features}</strong></li>
      <li>Jumlah data train: <strong>${m.train_samples}</strong></li>
      <li>Jumlah data test: <strong>${m.test_samples}</strong></li>
      <li>MSE train: <strong>${m.mse_train.toFixed(4)}</strong></li>
      <li>MSE test: <strong>${m.mse_test.toFixed(4)}</strong></li>
      <li>RMSE test: <strong>${m.rmse_test.toFixed(4)}</strong></li>
      <li>Gap ratio: <strong>${(m.gap_ratio * 100).toFixed(1)}%</strong></li>
    </ul>

    <p>${summary}</p>

    <h4>Loss Curve</h4>
    <img src="${lossImg}" alt="MLP Training Loss"
         style="max-width:100%;border-radius:8px;margin-bottom:12px;" />

    <h4>Prediksi vs Aktual (Test)</h4>
    <img src="${scatterImg}" alt="Prediksi vs Aktual (MLP)"
         style="max-width:100%;border-radius:8px;margin-bottom:16px;" />

    <h4>Kesimpulan</h4>
    <p style="white-space: pre-line;">${conclusion}</p>
  `;

  showModal("Hasil Algoritma MLP", htmlResult);

  // Tambahan baru: render detail proses & filter grafik di website (bukan cuma modal)
  renderProcessSection("MLP", data);
}

function renderRfModal(data) {
  const m = data.metrics;
  const summary = data.summary;
  const conclusion = data.conclusion || "Backend tidak mengirim kesimpulan.";
  const scatterImg = data.plots.pred_vs_actual;
  const fiImg = data.plots.feature_importance;

  const htmlResult = `
    <p>Training selesai menggunakan <strong>Random Forest Regressor</strong> di backend Python.</p>
    <ul>
      <li>Jumlah fitur: <strong>${m.n_features}</strong></li>
      <li>Jumlah data train: <strong>${m.train_samples}</strong></li>
      <li>Jumlah data test: <strong>${m.test_samples}</strong></li>
      <li>MSE train: <strong>${m.mse_train.toFixed(4)}</strong></li>
      <li>MSE test: <strong>${m.mse_test.toFixed(4)}</strong></li>
      <li>RMSE test: <strong>${m.rmse_test.toFixed(4)}</strong></li>
      <li>Gap ratio: <strong>${(m.gap_ratio * 100).toFixed(1)}%</strong></li>
      <li>Fitur paling berpengaruh: <strong>${m.top_feature}</strong></li>
    </ul>

    <p>${summary}</p>

    <h4>Prediksi vs Aktual (Test)</h4>
    <img src="${scatterImg}" alt="Prediksi vs Aktual (Random Forest)"
         style="max-width:100%;border-radius:8px;margin-bottom:16px;" />

    <h4>Feature Importance</h4>
    <img src="${fiImg}" alt="Feature Importance Random Forest"
         style="max-width:100%;border-radius:8px;margin-bottom:16px;" />

    <h4>Kesimpulan</h4>
    <p style="white-space: pre-line;">${conclusion}</p>
  `;

  showModal("Hasil Algoritma Random Forest", htmlResult);

  // Tambahan baru: render detail proses & filter grafik di website
  renderProcessSection("RF", data);
}

/* ======================================================
   SECTION PROSES & VISUALISASI DETAIL (BARU)
====================================================== */

const processSection = document.getElementById("processSection");
const processMeta = document.getElementById("processMeta");
const processStepsEl = document.getElementById("processSteps");
const featureSelect = document.getElementById("featureSelect");
const featureHint = document.getElementById("featureHint");
const featureScatterWrap = document.getElementById("featureScatterWrap");
const featureTrendWrap = document.getElementById("featureTrendWrap");
const backendPlots = document.getElementById("backendPlots");
const algoConclusionEl = document.getElementById("algoConclusion");

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderProcessSection(algoName, data) {
  if (!processSection) return;

  // show section
  processSection.classList.remove("hidden");

  const m = data.metrics || {};
  const steps = data.steps || [];
  const plots = data.plots || {};
  const interactive = data.interactive || {};
  const testPoints = interactive.test_points || [];
  const featNames = (m.feature_names || []).slice(); // array

  processMeta.innerHTML = `
    <div style="padding:10px;border-radius:10px;border:1px solid rgba(255,255,255,.15);">
      <strong>Algoritma aktif:</strong> ${escapeHtml(algoName)}<br/>
      <strong>Target:</strong> ${escapeHtml(m.target_name || "-")}<br/>
      <strong>Train/Test:</strong> ${m.train_samples ?? "-"} / ${m.test_samples ?? "-"}<br/>
      <strong>MSE test:</strong> ${m.mse_test != null ? m.mse_test.toFixed(4) : "-"} |
      <strong>RMSE test:</strong> ${m.rmse_test != null ? m.rmse_test.toFixed(4) : "-"} |
      <strong>Gap:</strong> ${m.gap_ratio != null ? (m.gap_ratio * 100).toFixed(1) + "%" : "-"}
      ${m.top_feature ? `<br/><strong>Top Feature (RF):</strong> ${escapeHtml(m.top_feature)}` : ""}
    </div>
  `;

  // langkah proses
  processStepsEl.innerHTML = "";
  steps.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${escapeHtml(s.title)}</strong><br/><span style="opacity:.9">${escapeHtml(s.detail)}</span>`;
    processStepsEl.appendChild(li);
  });

  // backend plots (yang sudah ada)
  backendPlots.innerHTML = "";
  if (algoName === "MLP" && plots.loss_curve) {
    backendPlots.innerHTML += `
      <div style="margin:10px 0;">
        <h4 style="margin:6px 0;">Loss Curve</h4>
        <img src="${plots.loss_curve}" alt="Loss Curve" style="max-width:100%;border-radius:8px;" />
      </div>
    `;
  }
  if (plots.pred_vs_actual) {
    backendPlots.innerHTML += `
      <div style="margin:10px 0;">
        <h4 style="margin:6px 0;">Prediksi vs Aktual</h4>
        <img src="${plots.pred_vs_actual}" alt="Prediksi vs Aktual" style="max-width:100%;border-radius:8px;" />
      </div>
    `;
  }
  if (algoName === "RF" && plots.feature_importance) {
    backendPlots.innerHTML += `
      <div style="margin:10px 0;">
        <h4 style="margin:6px 0;">Feature Importance</h4>
        <img src="${plots.feature_importance}" alt="Feature Importance" style="max-width:100%;border-radius:8px;" />
      </div>
    `;
  }

  // kesimpulan algo
  algoConclusionEl.innerHTML = `
    <div style="padding:10px;border-radius:10px;border:1px solid rgba(255,255,255,.15);line-height:1.5;">
      <div style="margin-bottom:8px;"><strong>Summary:</strong> ${escapeHtml(data.summary || "-")}</div>
      <div><strong>Conclusion:</strong><br/><span style="white-space:pre-line">${escapeHtml(data.conclusion || "-")}</span></div>
    </div>
  `;

  // setup filter fitur
  renderFeatureSelector(featNames, testPoints, m.target_name || "target");

  // scroll
  processSection.scrollIntoView({ behavior: "smooth" });
}

function renderFeatureSelector(featureNames, testPoints, targetName) {
  if (!featureSelect) return;

  // fallback kalau backend belum kirim
  if (!Array.isArray(featureNames) || featureNames.length === 0) {
    featureSelect.innerHTML = `<option value="">(fitur tidak tersedia)</option>`;
    featureHint.textContent = "";
    featureScatterWrap.innerHTML = "";
    featureTrendWrap.innerHTML = "";
    return;
  }

  // isi dropdown
  featureSelect.innerHTML = featureNames
    .map((fn, i) => `<option value="${escapeHtml(fn)}">${i + 1}. ${escapeHtml(fn)}</option>`)
    .join("");

  featureHint.textContent = `Pilih fitur untuk melihat korelasi dengan ${targetName} (dibatasi max ${testPoints.length} titik dari test).`;

  const initial = featureNames[0];
  drawInteractiveCharts(initial, testPoints);

  featureSelect.onchange = () => {
    const fn = featureSelect.value;
    drawInteractiveCharts(fn, testPoints);
  };
}

function drawInteractiveCharts(featureName, testPoints) {
  if (!featureName) return;

  // ambil (x=fitur, ytrue, ypred)
  const pts = [];
  for (const p of testPoints) {
    const x = p.features?.[featureName];
    const yt = p.y_true;
    const yp = p.y_pred;
    if (x == null || !Number.isFinite(x) || !Number.isFinite(yt) || !Number.isFinite(yp)) continue;
    pts.push({ x, yt, yp });
  }

  if (pts.length < 5) {
    featureScatterWrap.innerHTML = `<p style="opacity:.9;">Data tidak cukup untuk menggambar grafik dari fitur ini.</p>`;
    featureTrendWrap.innerHTML = "";
    return;
  }

  featureScatterWrap.innerHTML = renderSvgScatter(pts, featureName);
  featureTrendWrap.innerHTML = renderSvgTrend(pts, featureName);
}

function minMax(arr, key) {
  let mn = Infinity, mx = -Infinity;
  for (const o of arr) {
    const v = o[key];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx) || mn === mx) {
    // biar gak div 0
    mn = mn === Infinity ? 0 : mn - 1;
    mx = mx === -Infinity ? 1 : mx + 1;
  }
  return [mn, mx];
}

// SVG Scatter: x=feature, y=target (tampilkan 2 layer: y_true dan y_pred)
function renderSvgScatter(pts, featureName) {
  const W = 820, H = 320;
  const pad = 35;

  const [xmin, xmax] = minMax(pts, "x");
  const [yminA, ymaxA] = minMax(pts, "yt");
  const [yminP, ymaxP] = minMax(pts, "yp");
  const ymin = Math.min(yminA, yminP);
  const ymax = Math.max(ymaxA, ymaxP);

  const sx = (x) => pad + ((x - xmin) / (xmax - xmin)) * (W - pad * 2);
  const sy = (y) => (H - pad) - ((y - ymin) / (ymax - ymin)) * (H - pad * 2);

  const dotsTrue = pts.map(p => `<circle cx="${sx(p.x).toFixed(2)}" cy="${sy(p.yt).toFixed(2)}" r="2.4" opacity="0.55"></circle>`).join("");
  const dotsPred = pts.map(p => `<circle cx="${sx(p.x).toFixed(2)}" cy="${sy(p.yp).toFixed(2)}" r="2.4" opacity="0.55"></circle>`).join("");

  return `
    <div style="overflow:auto; border:1px solid rgba(255,255,255,.15); border-radius:10px; padding:8px;">
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:6px;">
        <strong>${escapeHtml(featureName)}</strong>
        <span style="opacity:.85;">(Scatter) ● Actual & ● Predicted</span>
      </div>
      <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
        <!-- axes -->
        <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" opacity="0.5"></line>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" opacity="0.5"></line>

        <!-- labels -->
        <text x="${pad}" y="${pad - 10}" font-size="12" opacity="0.9">y (target)</text>
        <text x="${W - pad - 80}" y="${H - 8}" font-size="12" opacity="0.9">x (feature)</text>

        <!-- points (actual) -->
        <g>
          ${dotsTrue}
        </g>

        <!-- points (predicted) -->
        <g>
          ${dotsPred}
        </g>

        <!-- legend -->
        <g transform="translate(${pad + 10}, ${pad + 5})" opacity="0.9">
          <text x="0" y="0" font-size="12">Legend:</text>
          <text x="0" y="16" font-size="12">• Actual (y_true)</text>
          <text x="0" y="32" font-size="12">• Predicted (y_pred)</text>
        </g>
      </svg>
      <small style="display:block; opacity:.85;">
        Tips: kalau titik Predicted jauh dari Actual, berarti error besar pada rentang fitur itu.
      </small>
    </div>
  `;
}

// SVG Trend: urutkan berdasarkan fitur lalu gambar 2 garis (actual vs predicted)
function renderSvgTrend(pts, featureName) {
  const W = 820, H = 320;
  const pad = 35;

  const sorted = pts.slice().sort((a, b) => a.x - b.x);
  const [xmin, xmax] = minMax(sorted, "x");
  const [yminA, ymaxA] = minMax(sorted, "yt");
  const [yminP, ymaxP] = minMax(sorted, "yp");
  const ymin = Math.min(yminA, yminP);
  const ymax = Math.max(ymaxA, ymaxP);

  const sx = (x) => pad + ((x - xmin) / (xmax - xmin)) * (W - pad * 2);
  const sy = (y) => (H - pad) - ((y - ymin) / (ymax - ymin)) * (H - pad * 2);

  function toPath(key) {
    let d = "";
    for (let i = 0; i < sorted.length; i++) {
      const p = sorted[i];
      const x = sx(p.x), y = sy(p[key]);
      d += (i === 0 ? "M" : "L") + x.toFixed(2) + "," + y.toFixed(2) + " ";
    }
    return d.trim();
  }

  const pathTrue = toPath("yt");
  const pathPred = toPath("yp");

  return `
    <div style="overflow:auto; border:1px solid rgba(255,255,255,.15); border-radius:10px; padding:8px;">
      <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:6px;">
        <strong>${escapeHtml(featureName)}</strong>
        <span style="opacity:.85;">(Trend) garis Actual vs Predicted setelah data diurutkan berdasarkan fitur</span>
      </div>
      <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
        <!-- axes -->
        <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" opacity="0.5"></line>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" opacity="0.5"></line>

        <path d="${pathTrue}" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.65"></path>
        <path d="${pathPred}" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.35"></path>

        <g transform="translate(${pad + 10}, ${pad + 5})" opacity="0.9">
          <text x="0" y="0" font-size="12">Legend:</text>
          <text x="0" y="16" font-size="12">— Actual (y_true)</text>
          <text x="0" y="32" font-size="12">— Predicted (y_pred)</text>
        </g>
      </svg>
      <small style="display:block; opacity:.85;">
        Ini bukan time-series asli (karena kita tidak tahu kolom waktu). “Trend” dibuat dengan mengurutkan data berdasarkan fitur terpilih.
      </small>
    </div>
  `;
}

/* ======================================================
   Kesimpulan Gabungan (upgrade narasi)
====================================================== */

function showGlobalSummary() {
  if (!lastMlpResult || !lastRfResult) {
    alert("Silakan jalankan MLP dan Random Forest terlebih dahulu.");
    return;
  }

  const mMlp = lastMlpResult.metrics;
  const cMlp = lastMlpResult.conclusion || "";
  const mRf = lastRfResult.metrics;
  const cRf = lastRfResult.conclusion || "";

  // Bandingkan performa berdasarkan MSE test
  let winner = "Seimbang";
  if (mMlp.mse_test < mRf.mse_test) winner = "MLP";
  if (mMlp.mse_test > mRf.mse_test) winner = "Random Forest";

  const gapMlp = (mMlp.gap_ratio ?? 0) * 100;
  const gapRf = (mRf.gap_ratio ?? 0) * 100;

  const betterAlgo = (winner === "MLP")
    ? "Pada dataset ini, MLP menghasilkan MSE test lebih kecil dibanding Random Forest → performa prediksi MLP lebih baik."
    : (winner === "Random Forest")
      ? "Pada dataset ini, Random Forest menghasilkan MSE test lebih kecil dibanding MLP → performa prediksi Random Forest lebih baik."
      : "Pada dataset ini, nilai MSE test MLP dan Random Forest relatif sama → performanya sebanding.";

  const overfitNote = `
    <ul>
      <li><strong>Gap MLP:</strong> ${gapMlp.toFixed(1)}% (selisih MSE train vs test → indikasi over/underfitting)</li>
      <li><strong>Gap RF:</strong> ${gapRf.toFixed(1)}% (selisih MSE train vs test → indikasi over/underfitting)</li>
    </ul>
  `;

  const html = `
    <h3>Perbandingan Utama</h3>
    <ul>
      <li><strong>Pemenang berdasarkan MSE test:</strong> ${winner}</li>
      <li><strong>MSE test:</strong> MLP=${mMlp.mse_test.toFixed(4)} vs RF=${mRf.mse_test.toFixed(4)}</li>
      <li><strong>RMSE test:</strong> MLP=${mMlp.rmse_test.toFixed(4)} vs RF=${mRf.rmse_test.toFixed(4)}</li>
    </ul>

    <h3>Ringkasan MLP</h3>
    <ul>
      <li>Jumlah fitur: <strong>${mMlp.n_features}</strong></li>
      <li>Data train/test: <strong>${mMlp.train_samples}</strong> / <strong>${mMlp.test_samples}</strong></li>
      <li>MSE train/test: <strong>${mMlp.mse_train.toFixed(4)}</strong> / <strong>${mMlp.mse_test.toFixed(4)}</strong></li>
      <li>RMSE train/test: <strong>${mMlp.rmse_train.toFixed(4)}</strong> / <strong>${mMlp.rmse_test.toFixed(4)}</strong></li>
    </ul>
    <p style="white-space: pre-line;">${escapeHtml(cMlp)}</p>

    <hr style="margin: 16px 0; border-color: rgba(255,255,255,0.15);" />

    <h3>Ringkasan Random Forest</h3>
    <ul>
      <li>Jumlah fitur: <strong>${mRf.n_features}</strong></li>
      <li>Data train/test: <strong>${mRf.train_samples}</strong> / <strong>${mRf.test_samples}</strong></li>
      <li>MSE train/test: <strong>${mRf.mse_train.toFixed(4)}</strong> / <strong>${mRf.mse_test.toFixed(4)}</strong></li>
      <li>RMSE train/test: <strong>${mRf.rmse_train.toFixed(4)}</strong> / <strong>${mRf.rmse_test.toFixed(4)}</strong></li>
      <li>Fitur paling berpengaruh (RF): <strong>${escapeHtml(mRf.top_feature)}</strong></li>
    </ul>
    <p style="white-space: pre-line;">${escapeHtml(cRf)}</p>

    <hr style="margin: 16px 0; border-color: rgba(255,255,255,0.15);" />

    <h3>Analisis Tambahan</h3>
    <p>${escapeHtml(betterAlgo)}</p>
    <h4>Indikasi Overfitting / Generalisasi</h4>
    ${overfitNote}

    <p>
      Kesimpulan ini bisa kamu pakai untuk jelasin ke dosen:
      (1) siapa error paling kecil, (2) siapa gap train-test paling stabil,
      (3) Random Forest lebih mudah interpretasi lewat feature importance,
      sedangkan MLP butuh scaling dan lebih sensitif hyperparameter.
    </p>
  `;

  summaryContent.innerHTML = html;
  summarySection.classList.remove("hidden");
  summarySection.scrollIntoView({ behavior: "smooth" });
}

/* ======================================================
   Modal Helpers (tetap)
====================================================== */

function showModal(title, htmlContent) {
  modalTitle.textContent = title;
  modalBody.innerHTML = htmlContent;
  modalOverlay.classList.remove("hidden");
}

function hideModal() {
  modalOverlay.classList.add("hidden");
}

modalClose.addEventListener("click", hideModal);
modalOk.addEventListener("click", hideModal);

modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) hideModal();
});

