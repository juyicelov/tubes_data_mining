import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ================================
# KONFIGURASI
# ================================
st.set_page_config(
    page_title="Clustering & Prediksi Penyakit",
    layout="wide"
)

st.title("🩺 Clustering & Prediksi Penyakit Jantung")

# ================================
# UPLOAD DATA
# ================================
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("Dataset berhasil dimuat")
st.write("Jumlah data:", df.shape[0])

# ================================
# DETEKSI KOLOM TARGET
# ================================
possible_targets = ["target", "output", "HeartDisease", "disease"]

target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    st.error("❌ Kolom target penyakit tidak ditemukan")
    st.info("Pastikan ada kolom: target / output / HeartDisease")
    st.stop()

# ================================
# AMBIL DATA NUMERIK SAJA
# ================================
numeric_df = df.select_dtypes(include=np.number)

if target_col not in numeric_df.columns:
    st.error("❌ Target bukan numerik")
    st.stop()

# ================================
# CLUSTERING
# ================================
st.header("🔹 Clustering Pasien")

X_cluster = numeric_df.drop(columns=[target_col])

scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_cluster)

k = st.slider("Jumlah Cluster", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

numeric_df["Cluster"] = cluster_labels

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

numeric_df["PCA1"] = X_pca[:, 0]
numeric_df["PCA2"] = X_pca[:, 1]

fig, ax = plt.subplots()
scatter = ax.scatter(
    numeric_df["PCA1"],
    numeric_df["PCA2"],
    c=numeric_df["Cluster"],
)
ax.set_title("Visualisasi Cluster Pasien")
st.pyplot(fig)

# ================================
# PREDIKSI PENYAKIT
# ================================
st.header("🧠 Prediksi Penyakit")

X = numeric_df.drop(columns=[target_col, "Cluster", "PCA1", "PCA2"])
y = numeric_df[target_col]

scaler_pred = StandardScaler()
X_scaled = scaler_pred.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
st.metric("Akurasi Model", round(acc, 3))

# ================================
# INPUT DATA BARU
# ================================
st.subheader("🔍 Prediksi Pasien Baru")

input_data = []
for col in X.columns:
    val = st.number_input(col, value=0.0)
    input_data.append(val)

if st.button("Prediksi"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler_pred.transform(input_array)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("⚠️ Pasien berpotensi memiliki penyakit")
    else:
        st.success("✅ Pasien tidak memiliki penyakit")
