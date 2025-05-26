import streamlit as st
from config import tickers
from views import training_view, prediction_view

# Konfigurasi tampilan aplikasi
st.set_page_config(
    page_title="Prediksi Saham",          # Nama di tab browser
    page_icon="📈",                       # Ikon aplikasi di tab dan sidebar
    layout="wide",                        # Layout lebar
    initial_sidebar_state="expanded"     # Sidebar terbuka secara default
)

# Judul utama aplikasi
st.title("📈 Aplikasi Prediksi Saham (LSTM Multi-Step)")

# Pilihan mode: Training atau Prediction
mode = st.sidebar.selectbox("Pilih Mode", ["Training", "Prediction"])

# Mengambil daftar ticker dari config
ticker_list = [t["ticker"] for t in tickers]

# Menentukan index default untuk ticker BBRI
default_index = ticker_list.index("BBRI.JK")  # BBRI sebagai default

# Dropdown untuk memilih saham/ticker
ticker = st.sidebar.selectbox(
    "Pilih Ticker",
    ticker_list,
    index=default_index,  # Menentukan default ticker
    format_func=lambda x: next(t["name"] for t in tickers if t["ticker"] == x)  # Menampilkan nama bank, bukan hanya tickernya
)

# Input parameter untuk model
time_step = st.sidebar.number_input("Time Step", min_value=1, value=60)  # Jumlah langkah waktu historis
n_steps = st.sidebar.number_input("N Steps (Prediksi ke depan)", min_value=1, value=5)  # Jumlah langkah prediksi

# Sumber data yang digunakan
st.sidebar.write("Sumber Data: [Yahoo Finance](https://finance.yahoo.com/)")

# Menjalankan tampilan berdasarkan mode yang dipilih
if mode == "Training":
    training_view.show(ticker, time_step, n_steps)
elif mode == "Prediction":
    prediction_view.show(ticker, time_step, n_steps)
