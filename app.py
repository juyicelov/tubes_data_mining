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
st.subheader("📂 Upload Dataset Kaggle (CSV)")

uploaded_file = st.file_uploader(
    "Upload file Shopping Mall Customer Segmentation Data (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("⚠️ Silakan upload dataset CSV terlebih dahulu.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("✅ Dataset berhasil dimuat!")
st.write("Jumlah Data:", df.shape[0])
st.write(df.head())

# ================================
# PREPROCESSING
# ================================
st.subheader("⚙️ Preprocessing")

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# K-MEANS CLUSTERING
# ================================
st.subheader("🔹 K-Means Clustering")

k = st.slider("Jumlah Cluster (K)", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, df['Cluster'])
st.success(f"Silhouette Score: {silhouette:.3f}")

# ================================
# VISUALISASI
# ================================
st.subheader("📈 Visualisasi Hasil Clustering")

fig, ax = plt.subplots()
ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segmentation (K-Means)")
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
st.subheader("📝 Input Data Customer Baru")

age = st.number_input("Age", 18, 80, 30)
income = st.number_input("Annual Income (k$)", 10, 200, 50)
score = st.number_input("Spending Score (1–100)", 1, 100, 50)

if st.button("🔍 Prediksi Cluster"):
    new_data = np.array([[age, income, score]])
    new_data_scaled = scaler.tra
