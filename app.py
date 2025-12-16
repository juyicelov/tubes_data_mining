import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Lung Cancer Clustering",
    layout="wide"
)

st.title("🫁 Lung Cancer Clustering")
st.caption("K-Means & Logistic Regression dengan Input Data Baru")

# ================================
# UPLOAD DATA
# ================================
uploaded_file = st.file_uploader(
    "Upload Lung Cancer Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("Dataset berhasil dimuat")
st.write("Jumlah baris:", df.shape[0])

# ================================
# TARGET & FEATURES
# ================================
TARGET = "PULMONARY_DISEASE"

if TARGET not in df.columns:
    st.error("Kolom PULMONARY_DISEASE tidak ditemukan")
    st.stop()

# ambil fitur numerik saja
X = df.drop(columns=[TARGET])
X = X.select_dtypes(include=["int64", "float64"])

# buang missing value dan sinkronkan df
df_clean = df.loc[X.dropna().index].copy()
X = X.dropna()

# ================================
# SCALING
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# KMEANS
# ================================
st.subheader("🔧 K-Means")

k = st.slider("Jumlah Cluster", 2, 6, 3)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
df_clean["Cluster"] = clusters

sil = silhouette_score(X_scaled, clusters)
st.info(f"Silhouette Score: {sil:.3f}")

# ================================
# LOGISTIC REGRESSION
# ================================
st.subheader("🤖 Logistic Regression")

# pastikan cluster > 1 kelas
if len(np.unique(clusters)) < 2:
    st.warning("Cluster hanya 1 kelas, Logistic Regression dilewati")
else:
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

# ================================
# INPUT DATA BARU
# ================================
st.subheader("✍️ Input Data Baru")

input_data = []
for col in X.columns:
    val = st.number_input(
        col,
        value=float(X[col].mean())
    )
    input_data.append(val)

if st.button("Prediksi Cluster"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred_cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Pasien masuk ke Cluster {pred_cluster}")

# ================================
# PREVIEW DATA
# ================================
st.subheader("📄 Data + Cluster")
st.dataframe(
    df_clean[[*X.columns, TARGET, "Cluster"]].head(20)
)
