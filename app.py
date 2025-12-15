import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA

# ================================
# CONFIG HALAMAN
# ================================
st.set_page_config(
    page_title="Clustering & Logistic Regression",
    layout="wide"
)

st.title("📊 Clustering & Logistic Regression")
st.caption("Dataset: BankChurners (Customer Churn Analysis)")

# ================================
# UPLOAD DATASET
# ================================
uploaded_file = st.file_uploader(
    "Upload file BankChurners.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Silakan upload dataset terlebih dahulu")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset berhasil dimuat")
st.write("Jumlah Data:", df.shape)

# ================================
# PREPROCESSING
# ================================
df = df.drop(columns=["CLIENTNUM"])

# Encode target
df["Attrition_Flag"] = df["Attrition_Flag"].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})

# Encode categorical features
cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ================================
# CLUSTERING
# ================================
st.subheader("🔹 Clustering (K-Means)")

k = st.slider("Jumlah Cluster", 2, 6, 4)

X_cluster = df.drop(columns=["Attrition_Flag"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df["Cluster"])
st.write("Silhouette Score:", round(sil, 3))

# PCA untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"]
)
ax.set_title("Visualisasi Cluster (PCA 2D)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)

# ================================
# LOGISTIC REGRESSION
# ================================
st.subheader("🔹 Logistic Regression (Prediksi Churn)")

X = df.drop(columns=["Attrition_Flag"])
y = df["Attrition_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Akurasi Logistic Regression: {round(acc * 100, 2)}%")

# ================================
# TAMPILKAN DATA
# ================================
st.subheader("🔹 Contoh Data")
st.dataframe(df.head())
