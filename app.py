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

    # Encoding otomatis Yes/No → 1/0
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

    # Jika ada kolom target "LUNG_CANCER"
    if "lung_cancer" in df_clean.columns.str.lower():
        target_col = [c for c in df_clean.columns if c.lower() == "lung_cancer"][0]
    else:
        target_col = df_clean.columns[-1]

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    st.write("Target yang digunakan:", target_col)

    # =============================
    # STANDARDIZATION
    # =============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =============================
    # LINEAR REGRESSION (RISK SCORE)
    # =============================
    st.subheader("4️⃣ Linear Regression (Risk Score)")

    lr = LinearRegression()
    lr.fit(X_scaled, y)

    risk_score = lr.predict(X_scaled)

    df_clean["Risk_Score"] = risk_score

    st.write("Contoh Risk Score:")
    st.write(df_clean[["Risk_Score"]].head())

    # =============================
    # CLUSTERING
    # =============================
    st.subheader("5️⃣ Clustering Berdasarkan Risk Score")

    k = st.slider("Jumlah Cluster", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(df_clean[["Risk_Score"]])

    silhouette = silhouette_score(df_clean[["Risk_Score"]], df_clean["Cluster"])
    st.write("Silhouette Score:", round(silhouette, 3))

    # =============================
    # VISUALIZATION
    # =============================
    st.subheader("6️⃣ Visualisasi Clustering")

    fig, ax = plt.subplots()
    ax.scatter(
        df_clean["Risk_Score"],
        df_clean["Cluster"]
    )
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Cluster")
    ax.set_title("Clustering Berdasarkan Risk Score (Linear Regression)")

    st.pyplot(fig)

    # =============================
    # INTERPRETATION
    # =============================
    st.subheader("7️⃣ Interpretasi Cluster")

    cluster_summary = df_clean.groupby("Cluster")["Risk_Score"].agg(["min", "mean", "max"])
    st.write(cluster_summary)

else:
    st.warning("Silakan upload dataset Lung Cancer CSV terlebih dahulu.")
