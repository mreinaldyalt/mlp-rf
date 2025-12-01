# mlp_server.py
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
from sklearn.metrics import mean_squared_error

app = FastAPI()

# Izinkan semua domain mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return "data:image/png;base64," + img_b64

@app.post("/mlp")
async def run_mlp(file: UploadFile = File(...)):
    print("\n=== REQUEST MLP DITERIMA ===")

    try:
        # 1. Baca CSV
        content = await file.read()
        print("Ukuran file diterima:", len(content))

        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        print("CSV terbaca. Shape asli:", df.shape)

        # 2. Ambil hanya kolom numerik
        numeric = df.select_dtypes(include="number")
        print("Kolom numerik:", numeric.shape)

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
        print("Setelah hapus NaN awal:", X.shape)

        if X.shape[0] < 1000:
            return JSONResponse(
                {"error": "Minimal butuh 1000 baris data setelah pembersihan."},
                status_code=400,
            )

        # 5. Sampling
        SAMPLE_FRAC = 0.05
        df_sampled = pd.concat([X, y], axis=1).sample(frac=SAMPLE_FRAC, random_state=42)
        print(f"Sampling {SAMPLE_FRAC*100}% data →", df_sampled.shape)

        # 6. Pisahkan fitur & target lagi
        X = df_sampled.iloc[:, :-1]
        y = df_sampled.iloc[:, -1]

        # 7. BERSIHKAN NaN SEKALI LAGI (fix utama)
        mask2 = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask2]
        y = y[mask2]
        print("Setelah hapus NaN setelah sampling:", X.shape)

        # Kalau masih NaN → error langsung
        if X.isna().any().any() or y.isna().any():
            return JSONResponse(
                {"error": "Dataset masih mengandung NaN setelah preprocessing."},
                status_code=400,
            )

        # 8. Train-test split
        print("Split train-test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Train:", X_train.shape, "Test:", X_test.shape)

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
            verbose=True
        )

        model.fit(X_train_scaled, y_train)
        print("Training selesai.")

        # 11. Evaluasi
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        mse_train = float(mean_squared_error(y_train, y_train_pred))
        mse_test = float(mean_squared_error(y_test, y_test_pred))
        rmse_train = mse_train ** 0.5
        rmse_test = mse_test ** 0.5

        # Selisih untuk komentar overfitting/underfitting
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
                "sehingga masih ada indikasi overfitting/underfitting ringan yang bisa "
                "diperbaiki dengan tuning hyperparameter."
            )
        else:
            fit_comment = (
                "Perbedaan MSE antara data train dan test cukup besar (>50%), "
                "mengindikasikan model belum seimbang antara proses belajar dan generalisasi. "
                "Diperlukan perbaikan lebih lanjut (misalnya penyesuaian arsitektur, epoch, "
                "atau regularisasi)."
            )

        # 12. Grafik loss curve
        plt.figure(figsize=(5, 3))
        plt.plot(model.loss_curve_)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("MLP Training Loss")
        loss_curve_img = fig_to_base64()

        # Pred vs Actual
        plt.figure(figsize=(5, 3))
        plt.scatter(y_test, y_test_pred, alpha=0.4)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediksi vs Aktual")
        scatter_img = fig_to_base64()

        print("=== MLP SELESAI TANPA ERROR ===")

        # Ringkasan singkat (bisa disebut di awal presentasi)
        summary = (
            f"Model MLP dengan hidden layer {model.hidden_layer_sizes}, max_iter={model.max_iter}, "
            f"menggunakan {len(X_train)} data train dan {len(X_test)} data test. "
            f"Hasil evaluasi memberikan MSE train = {mse_train:.4f} (RMSE ≈ {rmse_train:.4f}) "
            f"dan MSE test = {mse_test:.4f} (RMSE ≈ {rmse_test:.4f})."
        )

        # Kesimpulan naratif yang bisa kamu bacakan
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
        print("ERROR:", e)
        return JSONResponse(
            {"error": f"Terjadi error backend MLP: {str(e)}"},
            status_code=500
        )
