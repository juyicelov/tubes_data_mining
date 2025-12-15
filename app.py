import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================================
# KONFIGURASI HALAMAN
# ======================================
st.set_page_config(
    page_title="Prediksi Cluster Transaksi Retail",
    layout="wide"
)

st.title("🛒 Clustering & Prediksi Transaksi Retail Indonesia")
st.caption("Aplikasi clustering menggunakan K-Means dan prediksi cluster data baru")

# ======================================
# LOAD DATASET
# ======================================
df = pd.read_csv("transaksi_retail_indonesia.csv")

st.subheader("📂 Dataset")
st.write("Jumlah data:", df.shape[0])
st.dataframe(df.head())

# ======================================
# FITUR CLUSTERING
# ======================================
fitur = [
    "Umur_Pelanggan",
    "Pendapatan_Bulanan",
    "Jumlah_Item",
    "Total_Belanja"
]

X = df[fitur]

# ======================================
# SCALING
# ======================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================
# PILIH JUMLAH CLUSTER
# ======================================
st.subheader("🔢 Pengaturan Clustering")
k = st.slider("Jumlah Cluster (K)", 2, 8, 4)

# ======================================
# MODEL KMEANS
# ======================================
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X_scaled)

# ======================================
# VISUALISASI CLUSTER
# ======================================
st.subheader("📊 Visualisasi Cluster")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    df["Pendapatan_Bulanan"],
    df["Total_Belanja"],
    c=df["Cluster"]
)

ax.set_xlabel("Pendapatan Bulanan")
ax.set_ylabel("Total Belanja")
ax.set_title("Hasil Clustering Transaksi Retail")

st.pyplot(fig)

# ======================================
# INPUT DATA BARU (PREDIKSI)
# ======================================
st.subheader("🔮 Prediksi Cluster Data Baru")

col1, col2 = st.columns(2)

with col1:
    umur = st.number_input("Umur Pelanggan", min_value=17, max_value=80, value=30)
    pendapatan = st.number_input(
        "Pendapatan Bulanan (Rp)",
        min_value=1000000,
        max_value=20000000,
        value=5000000,
        step=500000
    )

with col2:
    jumlah_item = st.number_input("Jumlah Item Dibeli", min_value=1, max_value=20, value=3)
    total_belanja = st.number_input(
        "Total Belanja (Rp)",
        min_value=10000,
        max_value=10000000,
        value=300000,
        step=50000
    )

# ======================================
# TOMBOL PREDIKSI
# ======================================
if st.button("🔍 Prediksi Cluster"):
    data_baru = np.array([[umur, pendapatan, jumlah_item, total_belanja]])
    data_baru_scaled = scaler.transform(data_baru)

    cluster_prediksi = model.predict(data_baru_scaled)[0]

    st.success(f"✅ Data transaksi ini termasuk ke dalam **Cluster {cluster_prediksi}**")

    # Interpretasi sederhana
    st.info(
        f"Cluster {cluster_prediksi} merepresentasikan kelompok transaksi "
        f"dengan karakteristik yang mirip berdasarkan umur, pendapatan, "
        f"jumlah item, dan total belanja."
    )

# ======================================
# DATA HASIL
# ======================================
st.subheader("📌 Data dengan Label Cluster")
st.dataframe(df.head(20))
