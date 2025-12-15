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
st.set_page_config(page_title="Clustering & Logistic Regression", layout="wide")
st.title("📊 Clustering & Logistic Regression – Customer Segmentation")

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("Shopping Mall Customer Segmentation Data.csv")
    return df

df = load_data()

st.subheader("📁 Dataset Preview")
st.write(df.head())
st.write("Jumlah Data:", df.shape[0])

# ================================
# PREPROCESSING
# ================================
st.subheader("⚙️ Preprocessing")

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# CLUSTERING (K-MEANS)
# ================================
st.subheader("🔹 K-Means Clustering")

k = st.slider("Pilih Jumlah Cluster (K)", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, df['Cluster'])

st.success(f"Silhouette Score: {sil_score:.3f}")

# ================================
# VISUALISASI CLUSTER
# ================================
st.subheader("📈 Visualisasi Clustering")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Clustering")
st.pyplot(fig)

# ================================
# LOGISTIC REGRESSION (CLUSTER PREDICTION)
# ================================
st.subheader("🤖 Logistic Regression Model")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df['Cluster'], test_size=0.2, random_state=42
)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.info(f"Akurasi Logistic Regression: {acc:.2f}")

# ================================
# INPUT DATA BARU
# ================================
st.subheader("📝 Input Data Customer Baru")

age = st.number_input("Age", 18, 80, 30)
income = st.number_input("Annual Income (k$)", 10, 150, 50)
score = st.number_input("Spending Score (1-100)", 1, 100, 50)

if st.button("🔍 Prediksi"):
    new_data = np.array([[age, income, score]])
    new_data_scaled = scaler.transform(new_data)

    cluster_pred = kmeans.predict(new_data_scaled)[0]
    logreg_pred = logreg.predict(new_data_scaled)[0]

    st.success(f"""
    🔹 Hasil Prediksi:
    - Cluster (K-Means): {cluster_pred}
    - Cluster (Logistic Regression): {logreg_pred}
    """)

# ================================
# DATA CLUSTER SUMMARY
# ================================
st.subheader("📊 Ringkasan Cluster")
st.write(df.groupby('Cluster')[features].mean())
