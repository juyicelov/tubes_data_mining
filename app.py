import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ================================
# LOAD MODEL & PREPROCESSING
# ================================
scaler = pickle.load(open("scaler.pkl", "rb"))
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
logreg = pickle.load(open("logreg_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Clustering & Logistic Regression",
    layout="wide"
)

st.title("📊 Clustering & Logistic Regression – Customer Segmentation")
st.caption(
    "Aplikasi Data Mining untuk segmentasi pelanggan menggunakan "
    "K-Means dan prediksi cluster dengan Logistic Regression"
)

# ================================
# UPLOAD DATASET (UNTUK VISUALISASI SAJA)
# ================================
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("⚠️ Silakan upload dataset CSV")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("✅ Dataset berhasil dimuat")
st.write("Jumlah baris data:", df.shape[0])
st.write(df.head())

# ================================
# AMBIL FITUR SESUAI MODEL
# ================================
X = df[features]
X_scaled = scaler.transform(X)

# ================================
# PREDIKSI CLUSTER DATASET
# ================================
df["Cluster"] = kmeans.predict(X_scaled)

# ================================
# VISUALISASI CLUSTER
# ================================
st.subheader("📈 Visualisasi Hasil Clustering")

if len(features) >= 2:
    fig, ax = plt.subplots()
    ax.scatter(
        df[features[0]],
        df[features[1]],
        c=df["Cluster"]
    )
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("Visualisasi Clustering (2D)")
    st.pyplot(fig)

# ================================
# INPUT DATA BARU
# ================================
st.subheader("📝 Input Data Customer Baru")
st.caption("Masukkan data pelanggan untuk memprediksi cluster")

input_data = []

for col in features:
    min_val = int(df[col].min())
    max_val = int(df[col].max())

    value = st.number_input(
        f"Masukkan {col}",
        min_value=min_val,
        max_value=max_val,
        step=1
    )

    input_data.append(value)

# ================================
# PREDIKSI CLUSTER DATA BARU
# ================================
if st.button("🔍 Prediksi Cluster"):
    new_data = np.array([input_data])
    new_data_scaled = scaler.transform(new_data)

    cluster_kmeans = kmeans.predict(new_data_scaled)[0]
    cluster_logreg = logreg.predict(new_data_scaled)[0]

    st.success("✅ Hasil Prediksi")
    st.write(f"• Cluster (K-Means): **{cluster_kmeans}**")
    st.write(f"• Cluster (Logistic Regression): **{cluster_logreg}**")

# ================================
# RINGKASAN CLUSTER
# ================================
st.subheader("📊 Ringkasan Rata-rata Tiap Cluster")
st.write(df.groupby("Cluster")[features].mean())
