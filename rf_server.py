# rf_server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = FastAPI()

# Izinkan seluruh front-end mengakses API ini
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

@app.post("/rf")
async def run_rf(file: UploadFile = File(...)):
    print("\n=== REQUEST RANDOM FOREST DITERIMA ===")

    try:
        # 1. Baca CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        print("CSV terbaca:", df.shape)

        # 2. Ambil hanya kolom numerik
        numeric = df.select_dtypes(include="number")
        print("Kolom numerik:", numeric.shape)

        if numeric.shape[1] < 2:
            return JSONResponse({"error": "Minimal butuh 2 kolom numerik."}, status_code=400)

        # 3. Fitur & target
        X = numeric.iloc[:, :-1]
        y = numeric.iloc[:, -1]

        # 4. Hapus baris NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        print("Setelah hapus NaN:", X.shape)

        if X.shape[0] < 1000:
            return JSONResponse({"error": "Minimal 1000 baris setelah pembersihan."}, status_code=400)

        # 5. Sampling 5%
        SAMPLE_FRAC = 0.05
        df_sampled = pd.concat([X, y], axis=1).sample(frac=SAMPLE_FRAC, random_state=42)
        X = df_sampled.iloc[:, :-1]
        y = df_sampled.iloc[:, -1]
        print("Setelah sampling:", X.shape)

        # 6. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Train:", X_train.shape, "Test:", X_test.shape)

        # 7. Train Random Forest
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1
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
        plt.figure(figsize=(5,3))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediksi vs Aktual (Random Forest)")
        scatter_img = fig_to_base64()

        # 10. Grafik Feature Importance
        importance = model.feature_importances_
        feature_names = X.columns.tolist()

        plt.figure(figsize=(6,4))
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

        # Interpretasi selisih
        gap_ratio = abs(mse_test - mse_train) / mse_train

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
        print("ERROR:", e)
        return JSONResponse({"error": f"RF error: {str(e)}"}, status_code=500)

