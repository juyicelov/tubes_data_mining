import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering & Prediksi Penyakit")

st.title("🩺 Clustering & Prediksi Penyakit (Streamlit)")

# ================================
# UPLOAD DATASET
# ================================
uploaded_file = st.file_uploader("Upload dataset CSV penyakit", type=["csv"])

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
target = st.selectbox("Pilih kolom penyakit (0/1)", df.columns)

X = df.drop(columns=[target]).values
y = df[target].values

# ================================
# NORMALISASI (MANUAL)
# ================================
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# ================================
# K-MEANS MANUAL
# ================================
st.header("🔹 Clustering Pasien")

k = st.slider("Jumlah Cluster", 2, 6, 3)

def kmeans_manual(X, k, iter=50):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

if st.button("Jalankan Clustering"):
    labels, centroids = kmeans_manual(X, k)

    df["Cluster"] = labels
    st.write("Jumlah data per cluster:")
    st.write(df["Cluster"].value_counts())

    # ================================
    # PCA MANUAL (2D)
    # ================================
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    idx = np.argsort(eig_vals)[::-1]
    W = eig_vecs[:, idx[:2]]
    X_pca = X_centered @ W

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    ax.set_title("Visualisasi Clustering (PCA)")
    st.pyplot(fig)

# ================================
# PREDIKSI PENYAKIT (NEAREST CENTROID)
# ================================
st.header("🧠 Prediksi Penyakit")

input_data = []
for col in df.drop(columns=[target]).columns:
    val = st.number_input(col, value=0.0)
    input_data.append(val)

if st.button("Prediksi Penyakit"):
    input_array = np.array(input_data)
    input_array = (input_array - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    distances = np.linalg.norm(centroids - input_array, axis=1)
    nearest_cluster = np.argmin(distances)

    cluster_targets = y[df["Cluster"] == nearest_cluster]
    prediction = int(cluster_targets.mean() >= 0.5)

    if prediction == 1:
        st.error("⚠️ Pasien berpotensi MEMILIKI penyakit")
    else:
        st.success("✅ Pasien TIDAK memiliki penyakit")
