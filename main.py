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
#  CORS FIX SESUAI ARAHAN GPT
# ================================================================
app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    # nanti tambahkan GitHub Pages:
    # "https://mreinaldyalt.github.io",
    # "https://mreinaldyalt.github.io/as",
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
#  ENDPOINT MLP
# ================================================================
@app.post("/mlp")
async def run_mlp(file: UploadFile = File(...)):
    print("\n=== REQUEST MLP DITERIMA ===")

    try:
        # 1. Baca CSV
        content = await file.read()
        print("Ukuran file diterima (MLP):", len(content))

        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        print("CSV terbaca (MLP). Shape asli:", df.shape)

        # 2. Ambil kolom numerik
        numeric = df.select_dtypes(include="number")
        print("Kolom numerik (MLP):", numeric.shape)

        if numeric.shape[1] < 2:
            return JSONResponse(
                {"error": "Minimal butuh 2 kolom numerik (fitur + target)."},
                status_code=400,
            )

        # 3. Fitur dan target
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

        # 5. Sampling 5%
        SAMPLE_FRAC = 0.05
        df_sampled = pd.concat([X, y], axis=1).sample(frac=SAMPLE_FRAC, random_state=42)
        print(f"Sampling {SAMPLE_FRAC*100}% data (MLP) →", df_sampled.shape)

        # 6. Pisahkan lagi
        X = df_sampled.iloc[:, :-1]
        y = df_sampled.iloc[:, -1]

        # 7. Hapus NaN lagi
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

        # 9. StandardScaler
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
            fit_comment = "Perbedaan kecil (<20%), model generalisasi baik."
        elif gap_ratio < 0.5:
            fit_comment = "Perbedaan sedang (20–50%), ada potensi over/underfitting."
        else:
            fit_comment = "Perbedaan besar (>50%), model tidak seimbang."

        # 12. Plot loss curve
        plt.figure(figsize=(5, 3))
        plt.plot(model.loss_curve_)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("MLP Training Loss")
        loss_curve_img = fig_to_base64()

        # 13. Plot Prediksi vs Aktual
        plt.figure(figsize=(5, 3))
        plt.scatter(y_test, y_test_pred, alpha=0.4)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediksi vs Aktual (MLP)")
        scatter_img = fig_to_base64()

        print("=== MLP SELESAI TANPA ERROR ===")

        summary = (
            f"Model MLP dengan hidden layer {model.hidden_layer_sizes}, max_iter={model.max_iter}, "
            f"menggunakan {len(X_train)} data train dan {len(X_test)} data test."
        )

        conclusion = (
            f"MSE train = {mse_train:.4f}, MSE test = {mse_test:.4f} "
            f"(selisih {gap_ratio*100:.1f}%). {fit_comment}"
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


# ================================================================
#  ENDPOINT RANDOM FOREST
# ================================================================
@app.post("/rf")
async def run_rf(file: UploadFile = File(...)):
    print("\n=== REQUEST RANDOM FOREST DITERIMA ===")

    try:
        # 1. Baca CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        print("CSV terbaca (RF):", df.shape)

        # 2. Kolom numerik
        numeric = df.select_dtypes(include="number")
        print("Kolom numerik (RF):", numeric.shape)

        if numeric.shape[1] < 2:
            return JSONResponse({"error": "Minimal 2 kolom numerik."}, status_code=400)

        # 3. Fitur & target
        X = numeric.iloc[:, :-1]
        y = numeric.iloc[:, -1]

        # 4. Hapus NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

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

        # 6. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 7. Train model RF
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

        # 9. Plot Prediksi vs Aktual
        plt.figure(figsize=(5, 3))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediksi vs Aktual (Random Forest)")
        scatter_img = fig_to_base64()

        # 10. Feature Importance
        importance = model.feature_importances_
        feature_names = X.columns.tolist()

        plt.figure(figsize=(6, 4))
        plt.barh(feature_names, importance)
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        fi_img = fig_to_base64()

        # 11. Summary + Conclusion
        gap_ratio = abs(mse_test - mse_train) / mse_train if mse_train > 0 else 0.0

        important_feature_name = feature_names[importance.argmax()]

        if gap_ratio < 0.2:
            fit_comment = "Model memiliki generalisasi yang baik."
        elif gap_ratio < 0.5:
            fit_comment = "Model cukup baik, dengan sedikit overfitting."
        else:
            fit_comment = "Model cenderung overfitting."

        summary = (
            f"Random Forest dengan {model.n_estimators} trees dilatih "
            f"menggunakan {len(X_train)} data train dan {len(X_test)} data test."
        )

        conclusion = (
            f"MSE train {mse_train:.4f}, MSE test {mse_test:.4f} (selisih {gap_ratio*100:.1f}%). "
            f"{fit_comment} Fitur paling berpengaruh: '{important_feature_name}'."
        )

        return {
            "metrics": {
                "n_features": int(len(feature_names)),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
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
