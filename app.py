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
# CONFIG
# ================================
st.set_page_config(
    page_title="Clustering & Logistic Regression",
    layout="wide"
)

st.title("📊 Clustering & Logistic Regression – Customer Segmentation")

# ================================
# UPLOAD DATASET
# ================================
st.subheader("📂 Upload Dataset CSV")

uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Silakan upload dataset terlebih dahulu")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("✅ Dataset berhasil dimuat")
st.write("Jumlah baris:", df.shape[0])
st.write("Kolom dataset:", list(df.columns))
st.write(df.head())

# ================================
# PILIH FITUR (ANTI ERROR)
# ================================
st.subheader("🧩 Pilih Fitur untuk Clustering")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) < 2:
    st.error("Dataset harus memiliki minimal 2 kolom numerik")
    st.stop()

features = st.multiselect(
    "Pilih minimal 2 kolom numerik:",
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
df['Cluster'] = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, df['Cluster'])
st.success(f"Silhouette Score: {silhouette:.3f}")

# ================================
# VISUALISASI
# ================================
st.subheader("📈 Visualisasi Clustering")

if len(features) >= 2:
    fig, ax = plt.subplots()
    ax.scatter(
        df[features[0]],
        df[features[1]],
        c=df['Cluster']
    )
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("Hasil Clustering")
    st.pyplot(fig)

# ================================
# LOGISTIC REGRESSION
# ================================
st.subheader("🤖 Logistic Regression")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df['Cluster'],
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
st.subheader("📝 Input Data Baru")

input_data = []
for col in features:
    value = st.number_input(f"Masukkan {col}", float(df[col].min()), float(df[col].max()))
    input_data.append(value)

if st.button("🔍 Prediksi Cluster"):
    new_data = np.array([input_data])
    new_data_scaled = scaler.transform(new_data)

    pred_kmeans = kmeans.predict(new_data_scaled)[0]
    pred_logreg = logreg.predict(new_data_scaled)[0]

    st.success(f"""
    ✅ Hasil Prediksi:
    - Cluster (K-Means): {pred_kmeans}
    - Cluster (Logistic Regression): {pred_logreg}
    """)

# ================================
# RINGKASAN CLUSTER
# ================================
st.subheader("📊 Rata-rata Setiap Cluster")
st.write(df.groupby('Cluster')[features].mean())
