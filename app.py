import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

# ======================================
# KONFIGURASI HALAMAN
# ======================================
st.set_page_config(
    page_title="Clustering Pelanggan",
    layout="wide"
)

st.title("📊 Clustering & Logistic Regression")
st.caption(
    "Aplikasi Data Mining untuk segmentasi pelanggan "
    "menggunakan K-Means dan prediksi cluster dengan Logistic Regression"
)

# ======================================
# UPLOAD DATASET
# ======================================
st.subheader("📂 Upload Dataset CSV")

uploaded_file = st.file_uploader(
    "Upload dataset CSV (memiliki kolom numerik)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("⚠️ Silakan upload dataset terlebih dahulu")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("✅ Dataset berhasil dimuat")
st.write("Jumlah data:", df.shape[0])
st.write("Kolom:", list(df.columns))
st.dataframe(df.head())

# ======================================
# PEMILIHAN FITUR NUMERIK
# ======================================
st.subheader("🧩 Pilih Fitur untuk Clustering")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) < 2:
    st.error("Dataset harus memiliki minimal 2 kolom numerik")
    st.stop()

features = st.multiselect(
    "Pilih minimal 2 fitur numerik:",
    numeric_columns,
    default=numeric_columns[:3]
)

if len(features) < 2:
    st.warning("⚠️ Pilih minimal 2 fitur numerik")
    st.stop()

X = df[features]

# ======================================
# STANDARDISASI DATA
# ======================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================
# K-MEANS CLUSTERING
# ======================================
st.subheader("🔹 K-Means Clustering")

k = st.slider(
    "Jumlah Cluster (K)",
    min_value=2,
    max_value=8,
    value=3
)

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, df["Cluster"])
st.success(f"Silhouette Score: {sil_score:.3f}")

# ======================================
# VISUALISASI CLUSTER
# ======================================
st.subheader("📈 Visualisasi Clustering (2D)")

fig, ax = plt.subplots()

ax.scatter(
    df[features[0]],
    df[features[1]],
    c=df["Cluster"]
)

ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_title("Visualisasi Hasil K-Means")

st.pyplot(fig)

# ======================================
# LOGISTIC REGRESSION
# ======================================
st.subheader("🤖 Logistic Regression (Prediksi Cluster)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df["Cluster"],
    test_size=0.2,
    random_state=42
)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.info(f"Akurasi Logistic Regression: {accuracy:.2f}")

# ======================================
# INPUT DATA BARU
# ======================================
st.subheader("📝 Prediksi Cluster Data Baru")

input_data = []

for col in features:
    value = st.number_input(
        f"Masukkan nilai {col}",
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        step=1.0
    )
    input_data.append(value)

# ======================================
# PREDIKSI
# ======================================
if st.button("🔍 Prediksi Cluster"):
    new_data = np.array([input_data])
    new_data_scaled = scaler.transform(new_data)

    cluster_kmeans = kmeans.predict(new_data_scaled)[0]
    cluster_logreg = logreg.predict(new_data_scaled)[0]

    st.success("✅ Hasil Prediksi")
    st.write(f"Cluster (K-Means): **{cluster_kmeans}**")
    st.write(f"Cluster (Logistic Regression): **{cluster_logreg}**")

# ======================================
# RINGKASAN CLUSTER
# ======================================
st.subheader("📊 Rata-rata Fitur Tiap Cluster")
st.dataframe(df.groupby("Cluster")[features].mean())
