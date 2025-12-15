import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(page_title="Clustering & Prediksi Penyakit Jantung")

st.title("🩺 Clustering & Prediksi Penyakit Jantung")
st.caption("Metode: K-Means Clustering & Prediksi Berbasis Nearest Cluster")

# ================================
# UPLOAD DATASET
# ================================
uploaded_file = st.file_uploader(
    "Upload dataset penyakit (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df = df.select_dtypes(include=np.number)

st.success("Dataset berhasil dimuat")
st.write("Jumlah data:", df.shape[0])
st.dataframe(df.head())

# ================================
# PILIH TARGET
# ================================
st.subheader("🎯 Target Penyakit")
target = st.selectbox(
    "Pilih kolom target penyakit (0 = tidak sakit, 1 = sakit)",
    df.columns
)

X = df.drop(columns=[target]).values
y = df[target].values

# ================================
# NORMALISASI DATA
# ================================
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

# ================================
# K-MEANS MANUAL
# ================================
st.header("🔹 Clustering Pasien")

k = st.slider("Jumlah Cluster", 2, 6, 3)

def kmeans_manual(X, k, max_iter=50):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

if "centroids" not in st.session_state:
    st.session_state.centroids = None
    st.session_state.labels = None

if st.button("🚀 Jalankan Clustering"):
    labels, centroids = kmeans_manual(X_norm, k)
    st.session_state.centroids = centroids
    st.session_state.labels = labels

    df["Cluster"] = labels
    st.write("Jumlah data per cluster:")
    st.write(df["Cluster"].value_counts())

    # ================================
    # PCA MANUAL (VISUALISASI)
    # ================================
    Xc = X_norm - X_norm.mean(axis=0)
    cov = np.cov(Xc.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    idx = np.argsort(eig_vals)[::-1]
    W = eig_vecs[:, idx[:2]]
    X_pca = Xc @ W

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    ax.set_title("Visualisasi Clustering Pasien (PCA)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

# ================================
# INPUT DATA PASIEN BARU
# ================================
st.header("🧪 Input Data Pasien Baru")

if st.session_state.centroids is None:
    st.warning("⚠️ Jalankan clustering terlebih dahulu")
    st.stop()

# Definisi fitur + keterangan
feature_info = {
    "age": ("Usia pasien (tahun)", 20, 90, 1),
    "sex": ("Jenis kelamin (0 = perempuan, 1 = laki-laki)", 0, 1, 1),
    "cp": ("Tipe nyeri dada (0–3)", 0, 3, 1),
    "trestbps": ("Tekanan darah saat istirahat (mmHg)", 80, 200, 1),
    "chol": ("Kadar kolesterol serum (mg/dL)", 100, 400, 1),
    "fbs": ("Gula darah puasa > 120 mg/dL (0 = tidak, 1 = ya)", 0, 1, 1),
    "restecg": ("Hasil elektrokardiografi (0–2)", 0, 2, 1),
    "thalach": ("Detak jantung maksimum tercapai", 70, 210, 1),
    "exang": ("Nyeri dada saat olahraga (0 = tidak, 1 = ya)", 0, 1, 1),
    "oldpeak": ("Depresi ST akibat olahraga", 0.0, 6.0, 0.1),
    "slope": ("Kemiringan segmen ST (0–2)", 0, 2, 1),
    "ca": ("Jumlah pembuluh besar (0–4)", 0, 4, 1),
    "thal": ("Kondisi thalassemia (0–3)", 0, 3, 1),
}

input_data = []

for col in df.drop(columns=[target]).columns:
    if col in feature_info:
        desc, min_v, max_v, step = feature_info[col]
        st.markdown(f"**{col}** — {desc}")
        val = st.number_input(
            label=f"Masukkan nilai {col}",
            min_value=min_v,
            max_value=max_v,
            step=step,
            value=min_v
        )
    else:
        st.markdown(f"**{col}** — fitur numerik")
        val = st.number_input(col, value=0.0)

    input_data.append(val)

# ================================
# PREDIKSI PENYAKIT
# ================================
if st.button("🔍 Prediksi Penyakit"):
    input_array = (np.array(input_data) - X_mean) / X_std
    distances = np.linalg.norm(st.session_state.centroids - input_array, axis=1)
    nearest_cluster = np.argmin(distances)

    cluster_targets = y[st.session_state.labels == nearest_cluster]
    prediction = int(cluster_targets.mean() >= 0.5)

    if prediction == 1:
        st.error("⚠️ Pasien berpotensi MEMILIKI penyakit jantung")
    else:
        st.success("✅ Pasien TIDAK memiliki penyakit jantung")
