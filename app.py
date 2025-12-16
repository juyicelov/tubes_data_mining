import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="KMeans & Logistic Regression",
    layout="wide"
)

st.title("📊 Clustering K-Means & Logistic Regression")
st.caption("Aplikasi Data Mining dengan Input Data Baru")

# ================================
# UPLOAD DATASET
# ================================
st.subheader("📂 Upload Dataset CSV")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Silakan upload dataset terlebih dahulu")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("✅ Dataset berhasil dimuat")
st.write("Jumlah baris:", df.shape[0])
st.write("Jumlah kolom:", df.shape[1])

# ================================
# PILIH KOLOM NUMERIK
# ================================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if len(num_cols) < 2:
    st.error("Dataset harus memiliki minimal 2 kolom numerik")
    st.stop()

st.subheader("📌 Kolom Numerik")
st.write(num_cols)

X = df[num_cols].dropna()

# ================================
# SCALING
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# KMEANS CLUSTERING
# ================================
st.subheader("🔧 Pengaturan K-Means")
k = st.slider("Jumlah Cluster (k)", 2, 6, 3)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
df.loc[X.index, "Cluster"] = clusters

# ================================
# EVALUASI CLUSTER
# ================================
sil_score = silhouette_score(X_scaled, clusters)
st.info(f"📈 Silhouette Score: **{sil_score:.3f}**")

# ================================
# LOGISTIC REGRESSION
# ================================
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

st.success(f"🎯 Akurasi Logistic Regression: **{acc:.3f}**")

# ================================
# INPUT DATA BARU
# ================================
st.subheader("✍️ Input Data Baru")

input_data = []
for col in num_cols:
    val = st.number_input(
        f"Masukkan nilai {col}",
        value=float(X[col].mean())
    )
    input_data.append(val)

if st.button("🔮 Prediksi Cluster"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    predicted_cluster = logreg.predict(input_scaled)[0]

    st.success(f"✅ Data termasuk ke **Cluster {predicted_cluster}**")

# ================================
# TAMPILKAN DATA
# ================================
st.subheader("📄 Contoh Data + Cluster")
st.dataframe(df.head(20))
