# =========================================
# CLUSTERING & PREDIKSI PENYAKIT JANTUNG
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================================
# 1. LOAD DATASET
# =========================================
df = pd.read_csv("heart_disease_dataset.csv")

print("Jumlah data:", df.shape)
print(df.head())

# =========================================
# 2. CLUSTERING (UNSUPERVISED)
# =========================================
X_cluster = df.drop(columns=["target"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

print("\nHasil clustering:")
print(df["Cluster"].value_counts())

# =========================================
# 3. VISUALISASI CLUSTER (PCA)
# =========================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Visualisasi Clustering Pasien")
plt.show()

# =========================================
# 4. PREDIKSI PENYAKIT (SUPERVISED)
# =========================================
X = df.drop(columns=["target", "Cluster"])
y = df["target"]

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAkurasi Model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================
# 5. PREDIKSI DATA BARU
# =========================================
data_baru = np.array([[52,1,0,125,212,0,1,168,0,1.0,1,2,3]])
data_baru_scaled = scaler.transform(data_baru)

hasil = model.predict(data_baru_scaled)

if hasil[0] == 1:
    print("\nPasien berpotensi MEMILIKI penyakit jantung")
else:
    print("\nPasien TIDAK memiliki penyakit jantung")
