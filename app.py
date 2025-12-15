import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering & Prediksi Penyakit")

st.title("🩺 Clustering & Prediksi Penyakit")

uploaded_file = st.file_uploader("Upload dataset CSV penyakit", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df = df.select_dtypes(include=np.number)

st.success("Dataset berhasil dimuat")
st.write("Jumlah data:", df.shape[0])
st.dataframe(df.head())

target = st.selectbox("Pilih kolom penyakit (0/1)", df.columns)

X = df.drop(columns=[target]).values
y = df[target].values

# Normalisasi manual
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X = (X - X_mean) / X_std

st.header("🔹 Clustering Pasien")

k = st.slider("Jumlah Cluster", 2, 6, 3)

def kmeans_manual(X, k, iters=50):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(iters):
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

if st.button("Jalankan Clustering"):
    labels, centroids = kmeans_manual(X, k)
    st.session_state.centroids = centroids
    st.session_state.labels = labels

    df["Cluster"] = labels
    st.write("Jumlah data per cluster:")
    st.write(df["Cluster"].value_counts())

    # PCA manual
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    idx = np.argsort(eig_vals)[::-1]
    W = eig_vecs[:, idx[:2]]
    X_pca = Xc @ W

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    ax.set_title("Visualisasi Clustering (PCA)")
    st.pyplot(fig)

st.header("🧠 Prediksi Penyakit")

if st.session_state.centroids is None:
    st.warning("⚠️ Jalankan clustering terlebih dahulu")
    st.stop()

input_data = []
for col in df.drop(columns=[target]).columns:
    input_data.append(st.number_input(col, value=0.0))

if st.button("Prediksi Penyakit"):
    input_array = (np.array(input_data) - X_mean) / X_std
    distances = np.linalg.norm(st.session_state.centroids - input_array, axis=1)
    nearest_cluster = np.argmin(distances)

    cluster_targets = y[st.session_state.labels == nearest_cluster]
    prediction = int(cluster_targets.mean() >= 0.5)

    if prediction == 1:
        st.error("⚠️ Pasien berpotensi MEMILIKI penyakit")
    else:
        st.success("✅ Pasien TIDAK memiliki penyakit")
