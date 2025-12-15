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

st.set_page_config(page_title="Clustering & Prediksi Penyakit")

st.title("🩺 Clustering & Prediksi Penyakit Jantung")

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
numeric_df = df.select_dtypes(include=np.number)

# ================================
# PILIH TARGET
# ================================
target_col = st.selectbox("Pilih kolom target penyakit", numeric_df.columns)

# ================================
# CLUSTERING
# ================================
st.header("🔹 Clustering")

if st.button("Jalankan Clustering"):
    X_cluster = numeric_df.drop(columns=[target_col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    k = st.slider("Jumlah Cluster", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    ax.set_title("Visualisasi Cluster (PCA)")
    st.pyplot(fig)

# ================================
# PREDIKSI PENYAKIT
# ================================
st.header("🧠 Prediksi Penyakit")

if st.button("Latih Model & Prediksi"):
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Akurasi Model: {acc:.3f}")
