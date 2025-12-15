import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Lung Cancer Clustering", layout="centered")
st.title("🧬 Lung Cancer Risk Clustering (Simple)")

# =============================
# UPLOAD DATA
# =============================
uploaded_file = st.file_uploader("Upload Dataset Lung Cancer (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # =============================
    # PREPROCESSING SEDERHANA
    # =============================
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype(str).str.lower()
            if set(df_clean[col].unique()).issubset({"yes", "no"}):
                df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

    df_clean = df_clean.dropna()

    # =============================
    # PILIH TARGET
    # =============================
    target_col = "LUNG_CANCER" if "LUNG_CANCER" in df_clean.columns else df_clean.columns[-1]

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # =============================
    # TRAIN MODEL
    # =============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_scaled, y)

    df_clean["Risk_Score"] = lr.predict(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(df_clean[["Risk_Score"]])

    st.success("Model siap digunakan!")

    # =============================
    # INPUT SEDERHANA (HANYA 4)
    # =============================
    st.subheader("🧍 Input Data Pasien")

    age = st.slider("Usia", 10, 90, 40)
    smoking = st.selectbox("Merokok", [0, 1])
    alcohol = st.selectbox("Konsumsi Alkohol", [0, 1])
    chronic = st.selectbox("Penyakit Kronis", [0, 1])

    # Ambil rata-rata kolom lain
    input_data = X.mean().to_dict()
    input_data.update({
        X.columns[0]: age,
        X.columns[1]: smoking,
        X.columns[2]: alcohol,
        X.columns[3]: chronic
    })

    input_df = pd.DataFrame([input_data])

    # =============================
    # PREDIKSI
    # =============================
    if st.button("Prediksi Risiko"):
        input_scaled = scaler.transform(input_df)
        risk = lr.predict(input_scaled)[0]
        cluster = kmeans.predict([[risk]])[0]

        st.success(f"Risk Score: {risk:.2f}")
        st.info(f"Cluster Risiko: {cluster}")

        if cluster == 0:
            st.write("🟢 Risiko Rendah")
        elif cluster == 1:
            st.write("🟡 Risiko Sedang")
        else:
            st.write("🔴 Risiko Tinggi")

else:
    st.info("Silakan upload dataset untuk memulai.")
