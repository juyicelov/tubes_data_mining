import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

# ================================
# CONFIG HALAMAN
# ================================
st.set_page_config(
    page_title="Clustering & Logistic Regression",
    layout="wide"
)

st.title("📊 Clustering & Logistic Regression – Customer Segmentation")
st.caption("Aplikasi Data Mining untuk mengelompokkan pelanggan dan memprediksi cluster")

# ================================
# UPLOAD DATASET
# ================================
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload dataset CSV (contoh: Shopping Mall Customer Segmentation Data)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("⚠️ Silakan upload dataset terlebih dahulu")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("✅ Dataset berhasil dimuat")
st.write("Jumlah baris:", df.shape[0])
st.write("Kolom dataset:", list(df.columns))
st.write(df.head())

# ================================
# PILIH FITUR NUMERIK
# ================================
st.subheader("🧩 Pemilihan Fitur")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) < 2:
    st.error("Dataset harus memiliki minimal 2 kolom numerik")
    st.stop()

features = st.multiselect(
    "Pilih fitur numerik untuk clustering (disarankan 3 fitur):",
    numeric_columns,
    default=numeric_columns[:3]
)

if len(features) < 2:
    st.warning("⚠️ Pilih minimal 2 fitur")
    st.stop()

X = df[features]

# ================================
# SCALING
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# K-MEANS CLUSTERING
# ================================
st.subheader("🔹 K-Means Clustering")

k = st.slider("Jumlah Cluster (K)", 2, 8, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, df["Cluster"])
st.success(f"Silhouette Score: {sil_score:.3f}")

# ================================
# VISUALISASI CLUSTER
# ================================
st.subheader("📈 Visualisasi Hasil Clustering")

if len(features) >= 2:
    fig, ax = plt.subplots()
    ax.scatter(
        df[features[0]],
        df[features[1]],
        c=df["Cluster"]
    )
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("Visualisasi Clustering (2D)")
    st.pyplot(fig)

# ================================
# LOGISTIC REGRESSION
# ================================
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

# ================================
# INPUT DATA BARU
# ================================
st.subheader("📝 Input Data Customer Baru")
st.caption("Isi data berikut untuk memprediksi cluster pelanggan")

input_data = []

for col in features:
    min_val = int(df[col].min())
    max_val = int(df[col].max())

    if "age" in col.lower():
        help_text = "Usia pelanggan dalam satuan tahun"
    elif "income" in col.lower():
        help_text = "Pendapatan tahunan pelanggan (biasanya dalam ribuan)"
    elif "spending" in col.lower():
        help_text = "Skor tingkat pengeluaran (1 = rendah, 100 = tinggi)"
    else:
        help_text = "Nilai numerik untuk atribut ini"

    value = st.number_input(
        f"Masukkan {col}",
        min_value=min_val,
        max_value=max_val,
        step=1,
        help=help_text
    )

    input_data.append(value)

# ================================
# PREDIKSI
# ================================
if st.button("🔍 Prediksi Cluster"):
    new_data = np.array([input_data])
    new_data_scaled = scaler.transform(new_data)

    cluster_kmeans = kmeans.predict(new_data_scaled)[0]
    cluster_logreg = logreg.predict(new_data_scaled)[0]

    st.success("✅ Hasil Prediksi")
    st.write(f"• Cluster (K-Means): **{cluster_kmeans}**")
    st.write(f"• Cluster (Logistic Regression): **{cluster_logreg}**")

# ================================
# RINGKASAN CLUSTER
# ================================
st.subheader("📊 Ringkasan Rata-rata Tiap Cluster")
st.write(df.groupby("Cluster")[features].mean())
