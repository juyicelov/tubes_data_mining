import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Lung Cancer Clustering",
    layout="wide"
)

st.title("🫁 Lung Cancer Clustering")
st.caption("K-Means Clustering & Logistic Regression (Prediksi Cluster)")

# ==============================
# UPLOAD DATASET
# ==============================
uploaded_file = st.file_uploader(
    "Upload Lung Cancer Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload dataset CSV")
    st.stop()

# ==============================
# LOAD DATA (AMAN)
# ==============================
try:
    df = pd.read_csv(uploaded_file)
except Exception:
    df = pd.read_csv(uploaded_file, sep=";")

st.success("Dataset berhasil dimuat")
st.write("Shape data:", df.shape)

# ==============================
# TARGET & FEATURES
# ==============================
TARGET_COL = "PULMONARY_DISEASE"

if TARGET_COL not in df.columns:
    st.error("Kolom 'PULMONARY_DISEASE' tidak ditemukan")
    st.stop()

# Ambil fitur numerik saja
X = df.drop(columns=[TARGET_COL])
X = X.select_dtypes(include=["int64", "float64"])

if X.shape[1] == 0:
    st.error("Tidak ada kolom numerik untuk clustering")
    st.stop()

# Buang missing value dan sinkronkan index
X = X.dropna()
df_clean = df.loc[X.index].copy()

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# KMEANS
# ==============================
st.subheader("🔧 K-Means Clustering")

k = st.slider(
    "Jumlah Cluster (k)",
    min_value=2,
    max_value=6,
    value=3
)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
df_clean["Cluster"] = clusters

sil_score = silhouette_score(X_scaled, clusters)
st.info(f"Silhouette Score: {sil_score:.3f}")

# ==============================
# LOGISTIC REGRESSION
# ==============================
st.subheader("🤖 Logistic Regression (Prediksi Cluster)")

if len(np.unique(clusters)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        clusters,
        test_size=0.2,
        random_state=42
    )

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Akurasi Logistic Regression: {acc:.3f}")
else:
    st.warning("Cluster hanya satu kelas, Logistic Regression dilewati")

# ==============================
# INPUT DATA BARU (AMAN)
# ==============================
st.subheader("✍️ Input Data Pasien Baru")

input_values = []
for col in X.columns:
    input_values.append(
        st.slider(
            label=col,
            min_value=float(X[col].min()),
            max_value=float(X[col].max()),
            value=float(X[col].mean())
        )
    )

if st.button("Prediksi Cluster"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred_cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Pasien termasuk ke **Cluster {pred_cluster}**")

# ==============================
# DATA PREVIEW
# ==============================
st.subheader("📄 Contoh Data Hasil Clustering")
st.dataframe(
    df_clean[[*X.columns, TARGET_COL, "Cluster"]].head(20),
    use_container_width=True
)
