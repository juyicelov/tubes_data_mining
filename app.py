import streamlit as st
import pandas as pd

st.title("TEST CSV")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    st.success("CSV berhasil dibaca")
    st.write(df.head())
    st.write("Shape:", df.shape)
except Exception as e:
    st.error("Gagal membaca CSV")
    st.exception(e)
