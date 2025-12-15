import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering Transaksi Retail", layout="wide")
st.title("🛒 Clustering Transaksi Retail Indonesia")

df = pd.read_csv("transaksi_retail_indonesia.csv")

st.write("Jumlah data:", df.shape[0])
st.dataframe(df.head())

fitur = ["Umur_Pelanggan", "Pendapatan_Bulanan", "Jumlah_Item", "Total_Belanja"]
X = df[fitur]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Jumlah Cluster", 2, 8, 4)

model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X_scaled)

fig, ax = plt.subplots()
ax.scatter(df["Pendapatan_Bulanan"], df["Total_Belanja"], c=df["Cluster"])
ax.set_xlabel("Pendapatan Bulanan")
ax.set_ylabel("Total Belanja")
ax.set_title("Hasil Clustering Transaksi")

st.pyplot(fig)

st.subheader("Data dengan Cluster")
st.dataframe(df.head(20))
