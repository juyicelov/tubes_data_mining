import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================
# STREAMLIT CONFIG
# =============================
st.set_page_config(page_title="Lung Cancer Clustering", layout="wide")
st.title("🧬 Lung Cancer Clustering using Linear Regression")
st.write("Data Mining Project - Clustering berbasis Risk Score")

# =============================
# LOAD DATA
# =============================
st.subheader("1️⃣ Load Dataset")
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil dimuat!")
    st.write(df.head())

    # =============================
    # DATA PREPROCESSING
    # =============================
    st.subheader("2️⃣ Data Preprocessing")
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].astype(str).str.lower()
            if set(df_clean[col].unique()).issubset({"yes", "no"}):
                df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

    df_clean = df_clean.dropna()
    st.write("Data setelah preprocessing:")
    st.write(df_clean.head())

    # =============================
    # FEATURE & TARGET
    # =============================
    st.subheader("3️⃣ Feature Selection")

    if "lung_cancer" in df_clean.columns.str.lower():
        target_col = [c for c in df_clean.columns if c.lower() == "lung_cancer"][0]
    else:
        target_col = df_clean.columns[-1]

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    st.write("Target:", target_col)

    # =============================
    # STANDARDIZATION
    # =============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =============================
    # LINEAR REGRESSION
    # =============================
    st.subheader("4️⃣ Linear Regression (Risk Score)")
    lr = LinearRegression()
    lr.fit(X_scaled, y)

    df_clean["Risk_Score"] = lr.predict(X_scaled)

    # =============================
    # CLUSTERING
    # =============================
    st.subheader("5️⃣ Clustering")
    k = st.slider("Jumlah Cluster", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(df_clean[["Risk_Score"]])

    silhouette = silhouette_score(df_clean[["Risk_Score"]], df_clean["Cluster"])
    st.write("Silhouette Score:", round(silhouette, 3))

    # =============================
    # VISUALISASI
    # =============================
    st.subheader("6️⃣ Visualisasi Clustering")
    fig, ax = plt.subplots()
    ax.scatter(df_clean["Risk_Score"], df_clean["Cluster"])
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Cluster")
    st.pyplot(fig)

    # =============================
    # INPUT DATA BARU
    # =============================
    st.subheader("7️⃣ Input Data Pasien Baru")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

    input_df = pd.DataFrame([input_data])

    if st.button("Prediksi Risiko & Cluster"):
        input_scaled = scaler.transform(input_df)
        risk_pred = lr.predict(input_scaled)[0]
        cluster_pred = kmeans.predict([[risk_pred]])[0]

        st.success("✅ Prediksi Berhasil!")
        st.write(f"**Risk Score:** {risk_pred:.3f}")
        st.write(f"**Cluster Risiko:** {cluster_pred}")

        if cluster_pred == 0:
            st.info("Risiko Rendah")
        elif cluster_pred == 1:
            st.warning("Risiko Sedang")
        else:
            st.error("Risiko Tinggi")

    # =============================
    # INTERPRETASI CLUSTER
    # =============================
    st.subheader("8️⃣ Interpretasi Cluster")
    st.write(
        df_clean.groupby("Cluster")["Risk_Score"]
        .agg(["min", "mean", "max"])
    )

else:
    st.warning("Silakan upload dataset Lung Cancer CSV terlebih dahulu.")
