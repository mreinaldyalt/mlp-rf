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

  const reader = new FileReader();
  reader.onload = (event) => {
    const text = event.target.result;
    csvRawText = text; // simpan teks CSV
    renderCsvPreview(text);
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

// =======================
// BASE URL UNTUK BACKEND
// =======================
const BACKEND_BASE_URL = "https://mlp-rf.onrender.com";

const MLP_API_URL = `${BACKEND_BASE_URL}/mlp`;
const RF_API_URL  = `${BACKEND_BASE_URL}/rf`;

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
}

/* ======================================================
   Tombol Kesimpulan Gabungan
====================================================== */

function updateSummaryButtonState() {
  if (lastMlpResult && lastRfResult) {
    btnSummary.disabled = false;
    btnSummary.classList.add("enabled");
  } else {
    btnSummary.disabled = true;
    btnSummary.classList.remove("enabled");
  }
}

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
  let betterAlgo = "";
  if (mMlp.mse_test < mRf.mse_test) {
    betterAlgo = "Pada dataset ini, MLP menghasilkan nilai MSE test yang lebih kecil dibandingkan Random Forest, sehingga secara umum MLP memberikan performa prediksi yang lebih baik.";
  } else if (mMlp.mse_test > mRf.mse_test) {
    betterAlgo = "Pada dataset ini, Random Forest menghasilkan nilai MSE test yang lebih kecil dibandingkan MLP, sehingga secara umum Random Forest memberikan performa prediksi yang lebih baik.";
  } else {
    betterAlgo = "Pada dataset ini, nilai MSE test MLP dan Random Forest relatif sama, sehingga keduanya memiliki performa yang sebanding.";
  }

  const html = `
    <h3>Ringkasan MLP</h3>
    <ul>
      <li>Jumlah fitur: <strong>${mMlp.n_features}</strong></li>
      <li>Data train: <strong>${mMlp.train_samples}</strong></li>
      <li>Data test: <strong>${mMlp.test_samples}</strong></li>
      <li>MSE train: <strong>${mMlp.mse_train.toFixed(4)}</strong></li>
      <li>MSE test: <strong>${mMlp.mse_test.toFixed(4)}</strong></li>
      <li>RMSE train: <strong>${mMlp.rmse_train.toFixed(4)}</strong></li>
      <li>RMSE test: <strong>${mMlp.rmse_test.toFixed(4)}</strong></li>
    </ul>
    <p style="white-space: pre-line;">${cMlp}</p>

    <hr style="margin: 16px 0; border-color: rgba(255,255,255,0.15);" />

    <h3>Ringkasan Random Forest</h3>
    <ul>
      <li>Jumlah fitur: <strong>${mRf.n_features}</strong></li>
      <li>Data train: <strong>${mRf.train_samples}</strong></li>
      <li>Data test: <strong>${mRf.test_samples}</strong></li>
      <li>MSE train: <strong>${mRf.mse_train.toFixed(4)}</strong></li>
      <li>MSE test: <strong>${mRf.mse_test.toFixed(4)}</strong></li>
      <li>RMSE train: <strong>${mRf.rmse_train.toFixed(4)}</strong></li>
      <li>RMSE test: <strong>${mRf.rmse_test.toFixed(4)}</strong></li>
      <li>Fitur paling berpengaruh (RF): <strong>${mRf.top_feature}</strong></li>
    </ul>
    <p style="white-space: pre-line;">${cRf}</p>

    <hr style="margin: 16px 0; border-color: rgba(255,255,255,0.15);" />

    <h3>Perbandingan MLP vs Random Forest</h3>
    <p>${betterAlgo}</p>
    <p>
      Secara umum, perbandingan error (MSE/ RMSE) dan karakteristik model ini dapat dijadikan dasar
      untuk menjelaskan ke dosen mana algoritma yang lebih sesuai untuk kasus prediksi kelembapan
      udara pada dataset cuaca yang digunakan, serta bagaimana trade-off antara kompleksitas model dan akurasi.
    </p>
  `;

  summaryContent.innerHTML = html;
  summarySection.classList.remove("hidden");
  summarySection.scrollIntoView({ behavior: "smooth" });
}

/* ======================================================
   Dummy Output (fallback, tidak dipakai lagi)
====================================================== */

function generateDummyResult(algoName, csvText) {
  const rows = csvText.trim().split(/\r?\n/);
  const delimiter = rows.length > 0 ? detectDelimiter(rows[0]) : ",";
  const header = rows.length > 0 ? rows[0].split(delimiter) : [];
  const rowCount = Math.max(rows.length - 1, 0);

  return `
    <p>Ini adalah <strong>placeholder</strong> untuk algoritma <strong>${algoName}</strong>.</p>
    <ul>
      <li>Jumlah kolom (fitur): <strong>${header.length}</strong></li>
      <li>Jumlah baris data (tanpa header): <strong>${rowCount}</strong></li>
    </ul>
    <p>Nanti bagian ini bisa diganti dengan output algoritma yang sebenarnya.</p>
  `;
}

/* ======================================================
   Modal Helpers
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
