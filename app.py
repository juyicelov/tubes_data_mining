import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Clustering & Prediksi Penyakit Jantung",
    layout="wide"
)

st.title("🩺 Clustering & Prediksi Penyakit Jantung")
st.caption("Aplikasi Data Mining menggunakan K-Means & Logistic Regression")

# ================================
# UPLOAD DATASET
# ================================
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload file CSV (Heart Disease Dataset)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("⚠️ Silakan upload dataset terlebih dahulu")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("✅ Dataset berhasil dimuat")
st.write("Jumlah baris:", df.shape[0])
st.write("Jumlah kolom:", df.shape[1])

st.dataframe(df.head())

# ================================
# CLUSTERING
# ================================
st.subheader("🔹 Clustering Pasien (K-Means)")

X_cluster = df.drop(columns=['target'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

k = st.slider("Jumlah Cluster (K)", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

sil_score = silhouette_score(X_scaled, clusters)
st.write("📌 Silhouette Score:", round(sil_score, 3))

# ================================
# PCA VISUALISASI
# ================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

fig, ax = plt.subplots()
scatter = ax.scatter(
    df['PCA1'],
    df['PCA2'],
    c=df['Cluster'],
    cmap='viridis'
)
ax.set_title("Visualisasi Clustering Pasien (PCA)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
plt.colorbar(scatter)

st.pyplot(fig)

# ================================
# PREDIKSI PENYAKIT
# ================================
st.subheader("🧠 Prediksi Penyakit Jantung")

X = df.drop(columns=['target', 'Cluster', 'PCA1', 'PCA2'])
y = df['target']

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ================================
# EVALUASI MODEL
# ================================
st.markdown("### 📊 Evaluasi Model")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))

with col2:
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# ================================
# INPUT PASIEN BARU
# ================================
st.subheader("🧪 Prediksi Pasien Baru")

input_data = []

for col in X.columns:
    value = st.number_input(f"{col}", value=0.0)
    input_data.append(value)

if st.button("🔍 Prediksi Penyakit"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Pasien berpotensi MEMILIKI penyakit jantung")
    else:
        st.success("✅ Pasien TIDAK memiliki penyakit jantung")
