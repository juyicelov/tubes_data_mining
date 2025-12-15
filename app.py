import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# =================================================
# KONFIGURASI HALAMAN
# =================================================
st.set_page_config(page_title="Clustering Risiko Kanker Paru", layout="centered")
st.title("🫁 Clustering Risiko Kanker Paru-Paru")
st.write("Menggunakan Linear Regression dan K-Means Clustering")

# =================================================
# UPLOAD DATASET
# =================================================
st.header("1. Upload Dataset")
file = st.file_uploader("Upload file CSV Dataset Lung Cancer", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    st.success("Dataset berhasil diupload!")
    st.write("Contoh data:")
    st.write(data.head())

    # =================================================
    # PREPROCESSING DATA
    # =================================================
    st.header("2. Preprocessing Data")

    data_clean = data.copy()

    # Ubah Yes / No menjadi 1 / 0
    for kolom in data_clean.columns:
        if data_clean[kolom].dtype == object:
            data_clean[kolom] = data_clean[kolom].astype(str).str.lower()
            if set(data_clean[kolom].unique()).issubset({"yes", "no"}):
                data_clean[kolom] = data_clean[kolom].map({"yes": 1, "no": 0})

    # Hapus data kosong
    data_clean = data_clean.dropna()

    st.write("Data setelah preprocessing:")
    st.write(data_clean.head())

    # =================================================
    # MENENTUKAN FITUR & TARGET
    # =================================================
    st.header("3. Menentukan Fitur dan Target")

    if "LUNG_CANCER" in data_clean.columns:
        target = "LUNG_CANCER"
    else:
        target = data_clean.columns[-1]

    X = data_clean.drop(columns=[target])
    y = data_clean[target]

    st.write("Target yang digunakan:", target)

    # =================================================
    # STANDARISASI DATA
    # =================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =================================================
    # LINEAR REGRESSION → RISK SCORE
    # =================================================
    st.header("4. Perhitungan Risk Score")

    model_lr = LinearRegression()
    model_lr.fit(X_scaled, y)

    data_clean["Risk_Score"] = model_lr.predict(X_scaled)

    st.write("Contoh Risk Score:")
    st.write(data_clean[["Risk_Score"]].head())

    # =================================================
    # CLUSTERING
    # =================================================
    st.header("5. Clustering Risiko")

    model_kmeans = KMeans(n_clusters=3, random_state=42)
    data_clean["Cluster"] = model_kmeans.fit_predict(data_clean[["Risk_Score"]])

    st.write("Hasil Clustering:")
    st.write(data_clean[["Risk_Score", "Cluster"]].head())

    # =================================================
    # VISUALISASI
    # =================================================
    st.header("6. Visualisasi Clustering")

    fig, ax = plt.subplots()
    ax.scatter(data_clean["Risk_Score"], data_clean["Cluster"])
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Cluster")
    ax.set_title("Grafik Clustering Berdasarkan Risk Score")
    st.pyplot(fig)

    # =================================================
    # INPUT DATA PASIEN BARU (YA / TIDAK)
    # =================================================
    st.header("7. Input Data Pasien Baru")

    usia = st.slider("Usia", 10, 90, 40)

    merokok_text = st.selectbox("Merokok", ["Tidak", "Ya"])
    alkohol_text = st.selectbox("Konsumsi Alkohol", ["Tidak", "Ya"])
    penyakit_kronis_text = st.selectbox("Penyakit Kronis", ["Tidak", "Ya"])

    # Konversi ke angka
    merokok = 1 if merokok_text == "Ya" else 0
    alkohol = 1 if alkohol_text == "Ya" else 0
    penyakit_kronis = 1 if penyakit_kronis_text == "Ya" else 0

    # Isi data input (kolom lain pakai nilai rata-rata)
    input_data = X.mean().to_dict()
    kolom = list(X.columns)

    input_data[kolom[0]] = usia
    input_data[kolom[1]] = merokok
    input_data[kolom[2]] = alkohol
    input_data[kolom[3]] = penyakit_kronis

    input_df = pd.DataFrame([input_data])

    # =================================================
    # HASIL PREDIKSI
    # =================================================
    if st.button("Lihat Hasil Risiko"):
        input_scaled = scaler.transform(input_df)
        risk = model_lr.predict(input_scaled)[0]
        cluster = model_kmeans.predict([[risk]])[0]

        st.success(f"Risk Score: {risk:.2f}")
        st.info(f"Cluster Risiko: {cluster}")

        if cluster == 0:
            st.write("🟢 Risiko Rendah")
        elif cluster == 1:
            st.write("🟡 Risiko Sedang")
        else:
            st.write("🔴 Risiko Tinggi")

else:
    st.warning("Silakan upload dataset terlebih dahulu.")
