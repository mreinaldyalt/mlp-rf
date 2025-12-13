# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ================================================================
#  KONFIG: BATAS MAKSIMUM BARIS & CORS
# ================================================================
MAX_ROWS = 200_000          # maks baris yang dibaca dari CSV (untuk Render)
SAMPLE_FRAC = 0.02          # 2% sampling untuk training (lebih ringan)

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://mreinaldyalt.github.io",
    "https://mreinaldyalt.github.io/mlp-rf"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
#  FUNCTION: Convert matplotlib figure to Base64
# ================================================================
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return "data:image/png;base64," + img_b64


# ================================================================
#  TEMPLATE NARASI OTOMATIS (AGAR LAPORAN SELALU JELAS UNTUK DATASET APAPUN)
# ================================================================
def _fit_level(gap_ratio: float) -> str:
    if gap_ratio < 0.2:
        return "baik (generalisasi stabil)"
    if gap_ratio < 0.5:
        return "cukup (ada potensi over/underfitting)"
    return "buruk (indikasi overfitting/ketidakseimbangan tinggi)"


def build_report_narrative(
    algo_name: str,
    target_name: str,
    feature_names: list,
    train_samples: int,
    test_samples: int,
    mse_train: float,
    mse_test: float,
    rmse_test: float,
    gap_ratio: float,
    top_feature: str | None = None,
):
    """
    Narasi ini sengaja GENERIC, jadi bisa dipakai untuk dataset apa pun.
    Catatan penting:
    - Ini REGRESI: memprediksi nilai numerik target dari fitur numerik lain.
    - Ini BUKAN forecasting time-series kecuali split berdasarkan waktu.
    """
    feat_count = len(feature_names) if feature_names else 0
    fit_quality = _fit_level(gap_ratio)

    # Penjelasan arti RMSE dalam “satuan target”
    rmse_explain = (
        f"RMSE test {rmse_test:.4f} artinya rata-rata deviasi prediksi terhadap nilai aktual "
        f"sekitar ±{rmse_test:.4f} dalam satuan '{target_name}'."
    )

    # Tujuan program (jelas & universal)
    objective = (
        "Aplikasi ini melakukan pemodelan regresi untuk memprediksi variabel target numerik "
        "menggunakan fitur numerik lain, lalu membandingkan performa dua algoritma (MLP vs Random Forest) "
        "berdasarkan MSE/RMSE pada data uji dan stabilitas generalisasi (gap train-test)."
    )

    what_predicted = (
        f"Target yang diprediksi: '{target_name}'. "
        f"Jumlah fitur numerik yang digunakan: {feat_count}."
    )

    # Interpretasi kualitas generalisasi
    generalization = (
        f"Perbandingan train-test: MSE train={mse_train:.4f} vs MSE test={mse_test:.4f} "
        f"(gap {gap_ratio*100:.1f}%) → kualitas generalisasi: {fit_quality}."
    )

    # Rekomendasi keputusan (akurat vs stabil)
    decision = (
        "Jika tujuan utama adalah akurasi pada data uji (error sekecil mungkin), fokus ke model dengan RMSE/MSE test terendah. "
        "Jika tujuan utama adalah model yang stabil untuk data baru, fokus ke model dengan gap train-test lebih kecil."
    )

    # Saran perbaikan kalau gap besar
    tuning = []
    if gap_ratio >= 0.5:
        if algo_name.upper() == "RF":
            tuning.append("Random Forest terindikasi overfitting → coba batasi kompleksitas (mis. set max_depth, min_samples_leaf, min_samples_split) atau kurangi fitur yang tidak relevan.")
        if algo_name.upper() == "MLP":
            tuning.append("MLP terindikasi tidak stabil → coba naikkan max_iter, aktifkan early_stopping, atur alpha (regularisasi), dan tuning hidden layer/learning rate.")
    else:
        tuning.append("Model relatif stabil. Tuning tetap bisa dilakukan untuk menekan error test lebih jauh.")

    if top_feature:
        insight = f"Insight interpretasi (RF): fitur paling berpengaruh adalah '{top_feature}'."
    else:
        insight = "Insight interpretasi: MLP tidak punya feature importance langsung; interpretasi biasanya lewat analisis sensitivitas/SHAP (opsional)."

    # Kesimpulan otomatis untuk 1 model
    conclusion_text = (
        f"Model {algo_name} dilatih dengan {train_samples} data train dan {test_samples} data test. "
        f"{rmse_explain}\n"
        f"{generalization}\n"
        f"{insight}\n"
        f"Rekomendasi: {decision}\n"
        f"Catatan tuning: " + (" ".join(tuning))
    )

    return {
        "objective": objective,
        "what_predicted": what_predicted,
        "rmse_explain": rmse_explain,
        "generalization": generalization,
        "decision_guidance": decision,
        "insight": insight,
        "tuning_note": " ".join(tuning),
        "conclusion_text": conclusion_text,
    }

# ================================================================
#  ENDPOINT DATA PREPARATION
# ================================================================
@app.post("/prepare")
async def prepare_data(file: UploadFile = File(...)):
    steps = []

    def step(title, detail):
        steps.append({"title": title, "detail": detail})

    try:
        content = await file.read()
        df = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
            nrows=MAX_ROWS,
        )

        step("1) Load Dataset", f"Dataset awal: {df.shape[0]} baris × {df.shape[1]} kolom.")

        # Ambil numerik
        numeric = df.select_dtypes(include="number").copy()
        step("2) Seleksi Kolom Numerik", f"{numeric.shape[1]} kolom numerik terdeteksi.")

        # Duplikat
        before = numeric.shape[0]
        numeric = numeric.drop_duplicates()
        step(
            "3) Hapus Duplikasi",
            f"{before - numeric.shape[0]} baris duplikat dihapus."
        )

        # Missing value
        before = numeric.shape[0]
        numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)
        numeric = numeric.dropna()
        step(
            "4) Missing Value & Data Error",
            f"{before - numeric.shape[0]} baris mengandung NaN/Inf dihapus."
        )

        # Outlier (IQR — hanya laporan)
        Q1 = numeric.quantile(0.25)
        Q3 = numeric.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((numeric < (Q1 - 1.5 * IQR)) | (numeric > (Q3 + 1.5 * IQR)))
        outlier_rows = outlier_mask.any(axis=1).sum()

        step(
            "5) Deteksi Outlier (IQR)",
            f"Terdeteksi ±{outlier_rows} baris outlier (tidak dihapus, hanya ditandai)."
        )

        # Preview data bersih (10 baris)
        preview = numeric.head(10).to_dict(orient="records")

        return {
            "steps": steps,
            "clean_shape": numeric.shape,
            "preview": preview,
            "columns": numeric.columns.tolist(),
        }

    except Exception as e:
        return JSONResponse(
            {"error": f"Error data preparation: {str(e)}"},
            status_code=500,
        )


# ================================================================
#  ENDPOINT MLP
# ================================================================
@app.post("/mlp")
async def run_mlp(file: UploadFile = File(...)):
    print("\n=== REQUEST MLP DITERIMA ===")

    # step log untuk ditampilkan ke frontend
    steps = []
    def step(title: str, detail: str):
        steps.append({"title": title, "detail": detail})

    try:
        step("1) Verifikasi File", "Membaca file CSV yang diupload dan membatasi maksimal baris yang diproses.")

        # 1. Baca CSV (dibatasi MAX_ROWS)
        content = await file.read()
        file_size = len(content)
        print("Ukuran file diterima (MLP):", file_size)

        df = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
            nrows=MAX_ROWS,
        )
        step("2) Load Dataset", f"Dataset terbaca: {df.shape[0]} baris × {df.shape[1]} kolom (dibatasi MAX_ROWS={MAX_ROWS}).")

        # 2. Ambil kolom numerik
        numeric = df.select_dtypes(include="number").copy()
        step("3) Seleksi Kolom Numerik", f"Kolom numerik terdeteksi: {numeric.shape[1]} kolom.")

        if numeric.shape[1] < 2:
            return JSONResponse(
                {"error": "Minimal butuh 2 kolom numerik (fitur + target).", "steps": steps},
                status_code=400,
            )

        # 3. Cleansing: drop duplicate row
        before_dup = numeric.shape[0]
        numeric = numeric.drop_duplicates()
        removed_dup = before_dup - numeric.shape[0]
        step("4) Cleansing - Hapus Duplikat", f"Menghapus baris duplikat: {removed_dup} baris terhapus, sisa {numeric.shape[0]} baris.")

        # 4. Cleansing: replace inf/-inf → NaN lalu drop missing
        inf_count = int((~pd.isfinite(numeric)).sum().sum()) if hasattr(pd, "isfinite") else 0
        numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)

        # 5. Fitur dan target (kolom terakhir target)
        feature_names = numeric.columns[:-1].tolist()
        target_name = numeric.columns[-1]
        X = numeric.iloc[:, :-1]
        y = numeric.iloc[:, -1]

        # 6. Cleansing: missing value row removal
        before_nan = X.shape[0]
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        removed_nan = before_nan - X.shape[0]
        step(
            "5) Cleansing - Missing Value",
            f"Menghapus baris missing value (fitur/target): {removed_nan} baris terhapus, sisa {X.shape[0]} baris."
        )

        # 7. Cleansing: data error sederhana (nilai non-finite / kosong sudah ditangani).
        step(
            "6) Cleansing - Data Error",
            f"Penanganan nilai inf/-inf → NA (indikasi awal: {inf_count} sel). Baris invalid sudah tersaring pada tahap missing value."
        )

        if X.shape[0] < 1000:
            return JSONResponse(
                {"error": "Minimal butuh 1000 baris data setelah pembersihan (MLP).", "steps": steps},
                status_code=400,
            )

        # 8. Sampling
        df_sampled = pd.concat([X, y], axis=1).sample(
            frac=SAMPLE_FRAC,
            random_state=42,
        )
        step(
            "7) Sampling Dataset",
            f"Sampling {SAMPLE_FRAC*100:.1f}% untuk training lebih ringan: {df_sampled.shape[0]} baris terpilih."
        )

        X = df_sampled.iloc[:, :-1]
        y = df_sampled.iloc[:, -1]

        # 9. drop NaN lagi (jaga-jaga)
        before_nan2 = X.shape[0]
        mask2 = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask2]
        y = y[mask2]
        removed_nan2 = before_nan2 - X.shape[0]
        step("8) Validasi Akhir", f"Validasi ulang missing value setelah sampling: {removed_nan2} baris terhapus, sisa {X.shape[0]} baris.")

        if X.isna().any().any() or y.isna().any():
            return JSONResponse(
                {"error": "Dataset masih mengandung NaN setelah preprocessing (MLP).", "steps": steps},
                status_code=400,
            )

        # 10. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        step(
            "9) Split Train/Test",
            f"Train: {X_train.shape[0]} baris, Test: {X_test.shape[0]} baris (test_size=0.2)."
        )

        # 11. StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        step("10) Scaling (StandardScaler)", "Fitur distandarisasi (mean=0, std=1) untuk membantu training MLP.")

        # 12. Train MLP
        step("11) Training MLP", "Training model MLPRegressor (hidden layers 64-32, max_iter=20).")
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=20,
            batch_size=256,
            random_state=42,
            verbose=True,
        )
        model.fit(X_train_scaled, y_train)
        step("12) Training Selesai", "Model MLP selesai dilatih.")

        # 13. Evaluasi
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        mse_train = float(mean_squared_error(y_train, y_train_pred))
        mse_test = float(mean_squared_error(y_test, y_test_pred))
        rmse_train = mse_train ** 0.5
        rmse_test = mse_test ** 0.5

        gap = abs(mse_test - mse_train)
        gap_ratio = gap / mse_train if mse_train > 0 else 0.0

        if gap_ratio < 0.2:
            fit_comment = "Perbedaan kecil (<20%), model generalisasi baik."
        elif gap_ratio < 0.5:
            fit_comment = "Perbedaan sedang (20–50%), ada potensi over/underfitting."
        else:
            fit_comment = "Perbedaan besar (>50%), model tidak seimbang."

        step(
            "13) Evaluasi Model",
            f"MSE train={mse_train:.4f}, MSE test={mse_test:.4f}, RMSE test={rmse_test:.4f}, gap={gap_ratio*100:.1f}%. {fit_comment}"
        )

        # 14. Plot loss curve
        plt.figure(figsize=(5, 3))
        plt.plot(model.loss_curve_)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("MLP Training Loss")
        loss_curve_img = fig_to_base64()

        # 15. Plot Prediksi vs Aktual
        plt.figure(figsize=(5, 3))
        plt.scatter(y_test, y_test_pred, alpha=0.4)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediksi vs Aktual (MLP)")
        scatter_img = fig_to_base64()

        step("14) Visualisasi", "Menghasilkan grafik Loss Curve dan Prediksi vs Aktual.")

        # 16. Siapkan data ringan untuk filter grafik di frontend (batasi agar payload aman)
        # ambil max 500 titik dari test untuk interaktif
        max_points = 500
        test_df = X_test.copy()
        test_df["__y_true__"] = y_test.values
        test_df["__y_pred__"] = y_test_pred

        if len(test_df) > max_points:
            test_df = test_df.sample(n=max_points, random_state=42)

        test_points = []
        for _, row in test_df.iterrows():
            feats = {fn: (None if pd.isna(row[fn]) else float(row[fn])) for fn in feature_names}
            test_points.append({
                "features": feats,
                "y_true": float(row["__y_true__"]),
                "y_pred": float(row["__y_pred__"]),
            })

                        summary = (
            f"Model MLP (hidden layers={model.hidden_layer_sizes}, max_iter={model.max_iter}) "
            f"dilatih menggunakan {len(X_train)} data train dan {len(X_test)} data test "
            f"dari maksimum {MAX_ROWS} baris pertama dataset."
        )

        report = build_report_narrative(
            algo_name="MLP",
            target_name=str(target_name),
            feature_names=feature_names,
            train_samples=int(len(X_train)),
            test_samples=int(len(X_test)),
            mse_train=float(mse_train),
            mse_test=float(mse_test),
            rmse_test=float(rmse_test),
            gap_ratio=float(gap_ratio),
            top_feature=None,
        )

        conclusion = report["conclusion_text"]

                return {
            "steps": steps,
            "metrics": {
                "n_features": int(X.shape[1]),
                "feature_names": feature_names,
                "target_name": str(target_name),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "mse_train": mse_train,
                "mse_test": mse_test,
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
                "gap_ratio": float(gap_ratio),
            },
            "plots": {
                "loss_curve": loss_curve_img,
                "pred_vs_actual": scatter_img,
            },
            "interactive": {
                "test_points": test_points,
                "max_points": max_points,
            },
            "summary": summary,
            "conclusion": conclusion,
            "report": report,
        }
    except Exception as e:
        print("ERROR MLP:", e)
        return JSONResponse(
            {"error": f"Terjadi error backend MLP: {str(e)}", "steps": steps},
            status_code=500,
        )
        
# ================================================================
#  ENDPOINT RANDOM FOREST
# ================================================================
@app.post("/rf")
async def run_rf(file: UploadFile = File(...)):
    print("\n=== REQUEST RANDOM FOREST DITERIMA ===")

    steps = []
    def step(title: str, detail: str):
        steps.append({"title": title, "detail": detail})

    try:
        step("1) Verifikasi File", "Membaca file CSV yang diupload dan membatasi maksimal baris yang diproses.")

        content = await file.read()
        df = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
            nrows=MAX_ROWS,
        )
        step("2) Load Dataset", f"Dataset terbaca: {df.shape[0]} baris × {df.shape[1]} kolom (dibatasi MAX_ROWS={MAX_ROWS}).")

        numeric = df.select_dtypes(include="number").copy()
        step("3) Seleksi Kolom Numerik", f"Kolom numerik terdeteksi: {numeric.shape[1]} kolom.")

        if numeric.shape[1] < 2:
            return JSONResponse({"error": "Minimal 2 kolom numerik.", "steps": steps}, status_code=400)

        before_dup = numeric.shape[0]
        numeric = numeric.drop_duplicates()
        removed_dup = before_dup - numeric.shape[0]
        step("4) Cleansing - Hapus Duplikat", f"Menghapus baris duplikat: {removed_dup} baris terhapus, sisa {numeric.shape[0]} baris.")

        inf_count = int((~pd.isfinite(numeric)).sum().sum()) if hasattr(pd, "isfinite") else 0
        numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)

        feature_names = numeric.columns[:-1].tolist()
        target_name = numeric.columns[-1]
        X = numeric.iloc[:, :-1]
        y = numeric.iloc[:, -1]

        before_nan = X.shape[0]
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        removed_nan = before_nan - X.shape[0]
        step("5) Cleansing - Missing Value", f"Menghapus baris missing value (fitur/target): {removed_nan} baris terhapus, sisa {X.shape[0]} baris.")

        step("6) Cleansing - Data Error", f"Penanganan nilai inf/-inf → NA (indikasi awal: {inf_count} sel). Baris invalid tersaring pada tahap missing value.")

        if X.shape[0] < 1000:
            return JSONResponse(
                {"error": "Minimal 1000 baris setelah pembersihan (RF).", "steps": steps},
                status_code=400,
            )

        df_sampled = pd.concat([X, y], axis=1).sample(
            frac=SAMPLE_FRAC,
            random_state=42,
        )
        step("7) Sampling Dataset", f"Sampling {SAMPLE_FRAC*100:.1f}% untuk training lebih ringan: {df_sampled.shape[0]} baris terpilih.")

        X = df_sampled.iloc[:, :-1]
        y = df_sampled.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        step("8) Split Train/Test", f"Train: {X_train.shape[0]} baris, Test: {X_test.shape[0]} baris (test_size=0.2).")

        step("9) Training Random Forest", "Training RandomForestRegressor (n_estimators=150).")
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        step("10) Training Selesai", "Model Random Forest selesai dilatih.")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train = float(mean_squared_error(y_train, y_train_pred))
        mse_test = float(mean_squared_error(y_test, y_test_pred))
        rmse_train = mse_train ** 0.5
        rmse_test = mse_test ** 0.5

        gap_ratio = abs(mse_test - mse_train) / mse_train if mse_train > 0 else 0.0

        if gap_ratio < 0.2:
            fit_comment = "Model memiliki generalisasi yang baik."
        elif gap_ratio < 0.5:
            fit_comment = "Model cukup baik, dengan sedikit overfitting."
        else:
            fit_comment = "Model cenderung overfitting."

        step(
            "11) Evaluasi Model",
            f"MSE train={mse_train:.4f}, MSE test={mse_test:.4f}, RMSE test={rmse_test:.4f}, gap={gap_ratio*100:.1f}%. {fit_comment}"
        )

        plt.figure(figsize=(5, 3))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediksi vs Aktual (Random Forest)")
        scatter_img = fig_to_base64()

        importance = model.feature_importances_
        feature_names_full = X.columns.tolist()
        important_feature_name = feature_names_full[int(importance.argmax())]

        plt.figure(figsize=(6, 4))
        plt.barh(feature_names_full, importance)
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        fi_img = fig_to_base64()

        step("12) Visualisasi", "Menghasilkan grafik Prediksi vs Aktual dan Feature Importance.")

        # data ringan untuk filter grafik frontend
        max_points = 500
        test_df = X_test.copy()
        test_df["__y_true__"] = y_test.values
        test_df["__y_pred__"] = y_test_pred

        if len(test_df) > max_points:
            test_df = test_df.sample(n=max_points, random_state=42)

        test_points = []
        for _, row in test_df.iterrows():
            feats = {fn: (None if pd.isna(row[fn]) else float(row[fn])) for fn in feature_names_full}
            test_points.append({
                "features": feats,
                "y_true": float(row["__y_true__"]),
                "y_pred": float(row["__y_pred__"]),
            })

                summary = (
            f"Random Forest (n_estimators={model.n_estimators}) dilatih "
            f"menggunakan {len(X_train)} data train dan {len(X_test)} data test "
            f"dari maksimum {MAX_ROWS} baris pertama dataset."
        )

        report = build_report_narrative(
            algo_name="RF",
            target_name=str(target_name),
            feature_names=feature_names_full,
            train_samples=int(len(X_train)),
            test_samples=int(len(X_test)),
            mse_train=float(mse_train),
            mse_test=float(mse_test),
            rmse_test=float(rmse_test),
            gap_ratio=float(gap_ratio),
            top_feature=str(important_feature_name),
        )

        conclusion = report["conclusion_text"]


        return {
            "steps": steps,
            "metrics": {
                "n_features": int(len(feature_names_full)),
                "feature_names": feature_names_full,
                "target_name": str(target_name),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "mse_train": mse_train,
                "mse_test": mse_test,
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
                "gap_ratio": float(gap_ratio),
                "top_feature": important_feature_name,
            },
            "plots": {
                "pred_vs_actual": scatter_img,
                "feature_importance": fi_img,
            },
            "interactive": {
                "test_points": test_points,
                "max_points": max_points,
            },
                        "summary": summary,
            "conclusion": conclusion,
            "report": report,
        }

    except Exception as e:
        print("ERROR RF:", e)
        return JSONResponse(
            {"error": f"RF error: {str(e)}", "steps": steps},
            status_code=500,
        )
