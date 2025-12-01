// Modul sederhana untuk MLP 1-hidden-layer di sisi client.
// Tujuan utama: demonstrasi, bukan performa produksi.

(function (global) {
  function detectDelimiter(line) {
    const commaCount = (line.match(/,/g) || []).length;
    const semicolonCount = (line.match(/;/g) || []).length;
    return semicolonCount > commaCount ? ";" : ",";
  }

  // Parse CSV -> { X, y, nFeatures }
  // Asumsi: semua kolom numerik, kolom terakhir sebagai target.
  function parseCsvForMLP(csvText) {
    const rows = csvText.trim().split(/\r?\n/);
    if (rows.length < 3) {
      throw new Error("Data terlalu sedikit untuk dilatih.");
    }

    const delimiter = detectDelimiter(rows[0]);
    const header = rows[0].split(delimiter);
    const X = [];
    const y = [];

    for (let i = 1; i < rows.length; i++) {
      const row = rows[i].trim();
      if (!row) continue;

      const cols = row.split(delimiter);
      if (cols.length !== header.length) continue;

      const nums = cols.map((v) => parseFloat(v.replace(",", ".")));
      if (nums.some((v) => Number.isNaN(v))) {
        // jika ada non-numeric, skip baris
        continue;
      }

      const target = nums[nums.length - 1];
      const features = nums.slice(0, nums.length - 1);

      X.push(features);
      y.push(target);
    }

    if (X.length < 10) {
      throw new Error("Setelah pembersihan, data numerik kurang dari 10 baris.");
    }

    return { X, y, nFeatures: X[0].length };
  }

  // Normalisasi fitur: (x - mean) / std
  function computeNormalization(X) {
    const nSamples = X.length;
    const nFeat = X[0].length;
    const means = new Array(nFeat).fill(0);
    const stds = new Array(nFeat).fill(0);

    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeat; j++) {
        means[j] += X[i][j];
      }
    }
    for (let j = 0; j < nFeat; j++) {
      means[j] /= nSamples;
    }

    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeat; j++) {
        const diff = X[i][j] - means[j];
        stds[j] += diff * diff;
      }
    }
    for (let j = 0; j < nFeat; j++) {
      stds[j] = Math.sqrt(stds[j] / nSamples) || 1;
    }

    return { means, stds };
  }

  function applyNormalization(X, means, stds) {
    const nSamples = X.length;
    const nFeat = X[0].length;
    const Xn = new Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      const row = new Array(nFeat);
      for (let j = 0; j < nFeat; j++) {
        row[j] = (X[i][j] - means[j]) / stds[j];
      }
      Xn[i] = row;
    }

    return Xn;
  }

  // Inisialisasi bobot
  function initWeights(nInput, nHidden) {
    const W1 = new Array(nHidden);
    const b1 = new Array(nHidden).fill(0);
    const W2 = new Array(nHidden);
    let b2 = 0;

    const limit1 = Math.sqrt(2 / nInput);
    for (let j = 0; j < nHidden; j++) {
      W1[j] = new Array(nInput);
      for (let k = 0; k < nInput; k++) {
        W1[j][k] = (Math.random() * 2 - 1) * limit1;
      }
      W2[j] = (Math.random() * 2 - 1) * limit1;
    }

    return { W1, b1, W2, b2 };
  }

  // Forward pass untuk satu sample
  function forward(model, x) {
    const { W1, b1, W2, b2 } = model;
    const nHidden = W1.length;
    const nInput = x.length;

    const z1 = new Array(nHidden);
    const a1 = new Array(nHidden);

    for (let j = 0; j < nHidden; j++) {
      let sum = b1[j];
      const wj = W1[j];
      for (let k = 0; k < nInput; k++) {
        sum += wj[k] * x[k];
      }
      z1[j] = sum;
      a1[j] = sum > 0 ? sum : 0; // ReLU
    }

    let z2 = b2;
    for (let j = 0; j < nHidden; j++) {
      z2 += W2[j] * a1[j];
    }

    return { z1, a1, z2, yPred: z2 }; // output linear
  }

  // Training MLP (regresi) dengan SGD
  function trainMLP(X, y, options) {
    const nSamples = X.length;
    const nInput = X[0].length;
    const nHidden = options.hiddenUnits;
    const lr = options.learningRate;
    const epochs = options.epochs;

    const model = initWeights(nInput, nHidden);

    for (let epoch = 0; epoch < epochs; epoch++) {
      let mse = 0;

      for (let i = 0; i < nSamples; i++) {
        const x = X[i];
        const target = y[i];

        const { z1, a1, z2, yPred } = forward(model, x);
        const error = yPred - target;
        mse += error * error;

        const dOutput = 2 * error; // dL/dz2

        // grad W2, b2
        for (let j = 0; j < nHidden; j++) {
          model.W2[j] -= lr * dOutput * a1[j];
        }
        model.b2 -= lr * dOutput;

        // grad hidden
        const dHidden = new Array(nHidden);
        for (let j = 0; j < nHidden; j++) {
          let grad = dOutput * model.W2[j];
          if (z1[j] <= 0) grad = 0; // ReLU'
          dHidden[j] = grad;
        }

        // update W1, b1
        for (let j = 0; j < nHidden; j++) {
          const dH = dHidden[j];
          const wj = model.W1[j];
          for (let k = 0; k < nInput; k++) {
            wj[k] -= lr * dH * x[k];
          }
          model.b1[j] -= lr * dH;
        }
      }

      const epochMSE = mse / nSamples;
      // Jika ingin, bisa console.log:
      // console.log(`Epoch ${epoch + 1}/${epochs} - MSE: ${epochMSE.toFixed(4)}`);
    }

    return model;
  }

  function evaluateMLP(model, X, y) {
    const nSamples = X.length;
    let mse = 0;

    for (let i = 0; i < nSamples; i++) {
      const { yPred } = forward(model, X[i]);
      const error = yPred - y[i];
      mse += error * error;
    }

    return mse / nSamples;
  }

  // Fungsi utama dipanggil dari script.js
  function runMLPOnCsv(csvText) {
    try {
      const { X, y, nFeatures } = parseCsvForMLP(csvText);

      // split train/test 80/20
      const nSamples = X.length;
      const trainSize = Math.max(10, Math.floor(nSamples * 0.8));

      const Xtrain = X.slice(0, trainSize);
      const ytrain = y.slice(0, trainSize);
      const Xtest = X.slice(trainSize);
      const ytest = y.slice(trainSize);

      const { means, stds } = computeNormalization(Xtrain);
      const XtrainN = applyNormalization(Xtrain, means, stds);
      const XtestN = applyNormalization(Xtest, means, stds);

      const hiddenUnits = Math.min(16, Math.max(4, Math.round(nFeatures * 1.5)));
      const epochs = 40;
      const learningRate = 0.01;

      const model = trainMLP(XtrainN, ytrain, {
        hiddenUnits,
        learningRate,
        epochs,
      });

      const trainMSE = evaluateMLP(model, XtrainN, ytrain);
      const testMSE = evaluateMLP(model, XtestN, ytest);

      return {
        nFeatures,
        trainSamples: Xtrain.length,
        testSamples: Xtest.length,
        hiddenUnits,
        epochs,
        learningRate,
        trainMSE,
        testMSE,
      };
    } catch (err) {
      return { error: err.message || String(err) };
    }
  }

  global.MLPModule = {
    runMLPOnCsv,
  };
})(window);
