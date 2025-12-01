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

app = FastAPI()

# CORS: izinkan diakses dari domain mana saja (nanti bisa dipersempit ke domain GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fig_to_base64():
    """Render plot matplotlib aktif menjadi data URL base64 PNG."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return "data:image/png;base64," + img_b64


# ============================================================
#  Endpoint MLP
# ============================================================
@app.post("/mlp")
async def run_mlp(file: UploadFile = File(...)):
    print("\n=== REQUEST MLP DITERIMA ===")

    try:
        # 1. Baca CSV
        content = await file.read()
        print("Ukuran file diterima (MLP):", len(content))

        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        print("CSV terbaca (MLP). Shape asli:", df.shape)

        # 2. Ambil hanya kolom numerik
        numeric = df.select_dtypes(include="number")
        print("Kolom numerik (MLP):", numeric.shape)

        if numeric.shape[1] < 2:
            return JSONResponse(
                {"error": "Minimal butuh 2 kolom numerik (fitur + target)."},
                status_code=400,
            )

        # 3. Ambil fitur dan target
        X = numeric.iloc[:, :-1]
        y = numeric.iloc[:, -1]

        # 4. Hapus NaN penuh
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        print("Setelah hapus NaN awal (MLP):", X.shape)

        if X.shape[0] < 1000:
            return JSONResponse(
                {"error": "Minimal butuh 1000 baris data setelah pembersihan (MLP)."},
                status_code=400,
            )

        # 5. Sampling (5% supaya tidak terlalu berat)
        SAMPLE_FRAC = 0.05
        df_sampled = pd.concat([X, y], axis=1).sample(frac=SAMPLE_FRAC, random_state=42)
        print(f"Sampling {SAMPLE_FRAC*100}% data (MLP) →", df_sampled.shape)

        # 6. Pisahkan fitur & target lagi
        X = df_sampled.iloc[:, :-1]
        y = df_sampled.iloc[:, -1]

        # 7. Bersihkan NaN lagi
        mask2 = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask2]
        y = y[mask2]
        print("Setelah hapus NaN setelah sampling (MLP):", X.shape)

        if X.isna().any().any() or y.isna().any():
            return JSONResponse(
                {"error": "Dataset masih mengandung NaN setelah preprocessing (MLP)."},
                status_code=400,
            )

        # 8. Train-test split
        print("Split train-test (MLP)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Train (MLP):", X_train.shape, "Test:", X_test.shape)

        # 9. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 10. Train MLP
        print("Mulai training MLP...")
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
        print("Training MLP selesai.")

        # 11. Evaluasi
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        mse_train = float(mean_squared_error(y_train, y_train_pred))
        mse_test = float(mean_squared_error(y_test, y_test_pred))
        rmse_train = mse_train ** 0.5
        rmse_test = mse_test ** 0.5

        gap = abs(mse_test - mse_train)
        gap_ratio = gap / mse_train if mse_train > 0 else 0.0

        if gap_ratio < 0.2:
            fit_comment = (
                "Perbedaan MSE antara data train dan test relatif kecil (<20%), "
                "sehingga model cukup mampu melakukan generalisasi ke data baru."
            )
        elif gap_ratio < 0.5:
            fit_comment = (
                "Perbedaan MSE antara data train dan test berada di kisaran 20–50%, "
                "sehingga masih ada indikasi overfitting/underfitting ringan."
            )
        else:
            fit_comment = (
                "Perbedaan MSE antara data train dan test cukup besar (>50%), "
                "mengindikasikan model belum seimbang antara proses belajar dan generalisasi."
            )

        # 12. Grafik loss curve
        plt.figure(figsize=(5, 3))
        plt.plot(model.loss_curve_)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("MLP Training Loss")
        loss_curve_img = fig_to_base64()

        # 13. Grafik prediksi vs aktual
        plt.figure(figsize=(5, 3))
        plt.scatter(y_test, y_test_pred, alpha=0.4)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediksi vs Aktual (MLP)")
        scatter_img = fig_to_base64()

        print("=== MLP SELESAI TANPA ERROR ===")

        summary = (
            f"Model MLP dengan hidden layer {model.hidden_layer_sizes}, max_iter={model.max_iter}, "
            f"menggunakan {len(X_train)} data train dan {len(X_test)} data test. "
            f"Hasil evaluasi memberikan MSE train = {mse_train:.4f} (RMSE ≈ {rmse_train:.4f}) "
            f"dan MSE test = {mse_test:.4f} (RMSE ≈ {rmse_test:.4f})."
        )

        conclusion = (
            f"Dari hasil pengujian, nilai MSE train sebesar {mse_train:.4f} dan MSE test "
            f"sebesar {mse_test:.4f}, dengan selisih sekitar {gap_ratio*100:.1f}% relatif "
            f"terhadap MSE train. {fit_comment}\n\n"
            "Berdasarkan kurva loss yang menurun dan sebaran titik pada grafik prediksi vs aktual, "
            "model sudah mampu menangkap pola umum dari data target. Namun performa masih dapat "
            "ditingkatkan dengan melakukan tuning hyperparameter (jumlah neuron, learning rate, "
            "jumlah epoch) atau penambahan teknik regularisasi agar error pada data test dapat "
            "diturunkan lebih lanjut."
        )

        return {
            "metrics": {
                "n_features": int(X.shape[1]),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "mse_train": mse_train,
                "mse_test": mse_test,
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
            },
            "plots": {
                "loss_curve": loss_curve_img,
                "pred_vs_actual": scatter_img,
            },
            "summary": summary,
            "conclusion": conclusion,
        }

    except Exception as e:
        print("ERROR MLP:", e)
        return JSONResponse(
            {"error": f"Terjadi error backend MLP: {str(e)}"},
            status_code=500,
        )


# ============================================================
#  Endpoint Random Forest
# ============================================================
@app.post("/rf")
async def run_rf(file: UploadFile = File(...)):
    print("\n=== REQUEST RANDOM FOREST DITERIMA ===")

    try:
        # 1. Baca CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        print("CSV terbaca (RF):", df.shape)

        # 2. Ambil hanya kolom numerik
        numeric = df.select_dtypes(include="number")
        print("Kolom numerik (RF):", numeric.shape)

        if numeric.shape[1] < 2:
            return JSONResponse({"error": "Minimal butuh 2 kolom numerik."}, status_code=400)

        # 3. Fitur & target
        X = numeric.iloc[:, :-1]
        y = numeric.iloc[:, -1]

        # 4. Hapus baris NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        print("Setelah hapus NaN (RF):", X.shape)

        if X.shape[0] < 1000:
            return JSONResponse(
                {"error": "Minimal 1000 baris setelah pembersihan (RF)."},
                status_code=400,
            )

        # 5. Sampling 5%
        SAMPLE_FRAC = 0.05
        df_sampled = pd.concat([X, y], axis=1).sample(frac=SAMPLE_FRAC, random_state=42)
        X = df_sampled.iloc[:, :-1]
        y = df_sampled.iloc[:, -1]
        print("Setelah sampling (RF):", X.shape)

        # 6. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Train (RF):", X_train.shape, "Test:", X_test.shape)

        # 7. Train Random Forest
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # 8. Evaluasi
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train = float(mean_squared_error(y_train, y_train_pred))
        mse_test = float(mean_squared_error(y_test, y_test_pred))
        rmse_train = mse_train ** 0.5
        rmse_test = mse_test ** 0.5

        # 9. Grafik Prediksi vs Aktual
        plt.figure(figsize=(5, 3))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediksi vs Aktual (Random Forest)")
        scatter_img = fig_to_base64()

        # 10. Grafik Feature Importance
        importance = model.feature_importances_
        feature_names = X.columns.tolist()

        plt.figure(figsize=(6, 4))
        plt.barh(feature_names, importance)
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        fi_img = fig_to_base64()

        # 11. Summary + Conclusion
        summary = (
            f"Random Forest dengan {model.n_estimators} trees berhasil dilatih. "
            f"Jumlah fitur: {len(feature_names)}, data train: {len(X_train)}, test: {len(X_test)}. "
            f"MSE train = {mse_train:.4f}, MSE test = {mse_test:.4f}."
        )

        gap_ratio = abs(mse_test - mse_train) / mse_train if mse_train > 0 else 0.0

        if gap_ratio < 0.2:
            fit_comment = "Model memiliki generalisasi yang baik."
        elif gap_ratio < 0.5:
            fit_comment = "Model cukup baik, namun ada potensi overfitting ringan."
        else:
            fit_comment = "Model cenderung overfitting atau belum stabil."

        important_feature_name = feature_names[importance.argmax()]

        conclusion = (
            f"Hasil evaluasi menunjukkan MSE train {mse_train:.4f} dan MSE test {mse_test:.4f} "
            f"(RMSE test {rmse_test:.4f}). Selisih error sebesar {gap_ratio*100:.1f}% mengindikasikan bahwa "
            f"{fit_comment}.\n\n"
            f"Dari hasil perhitungan feature importance, variabel yang paling berpengaruh dalam prediksi adalah "
            f"'{important_feature_name}'. Hal ini menunjukkan bahwa variabel tersebut memiliki kontribusi terbesar "
            f"terhadap target yang diprediksi.\n\n"
            f"Secara keseluruhan, Random Forest mampu menangkap pola non-linear pada data cuaca dan memberikan "
            f"hasil prediksi yang cukup akurat. Performa dapat ditingkatkan dengan tuning jumlah tree, "
            f"max_depth, atau penyesuaian sampling data."
        )

        print("=== RANDOM FOREST SELESAI TANPA ERROR ===")

        return {
            "metrics": {
                "n_features": int(len(feature_names)),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_Test)) if False else int(len(X_test)),  # menjaga konsistensi
                "mse_train": mse_train,
                "mse_test": mse_test,
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
                "top_feature": important_feature_name,
            },
            "plots": {
                "pred_vs_actual": scatter_img,
                "feature_importance": fi_img,
            },
            "summary": summary,
            "conclusion": conclusion,
        }

    except Exception as e:
        print("ERROR RF:", e)
        return JSONResponse(
            {"error": f"RF error: {str(e)}"},
            status_code=500,
        )
