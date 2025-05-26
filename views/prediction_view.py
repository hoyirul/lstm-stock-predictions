import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import joblib
from data.refs import load_stock_data
from services.predictor import predict_future
from config import models_dir

def show(ticker, time_step, n_steps):
    # Menampilkan judul halaman prediksi
    st.header("🔮 Prediksi Harga Saham")

    # Memilih tanggal awal dan akhir data historis yang akan digunakan
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End Date", value=datetime.today())

    # Validasi tanggal
    if start_date > end_date:
        st.error("❌ Tanggal mulai tidak boleh lebih besar dari tanggal akhir.")
        return

    st.write(f"📅 Periode: {start_date} - {end_date}")
    
    # Komponen UI: loading bar dan status teks
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Tombol untuk memulai proses prediksi
    if st.button("📤 Load Model dan Prediksi"):
        # Cek keberadaan direktori dan file model
        ticker_dir = f"{models_dir}/{ticker.replace('.', '_')}"
        if not os.path.exists(ticker_dir):
            st.error(f"❌ Model atau scaler tidak ditemukan untuk ticker {ticker}. Silakan lakukan training terlebih dahulu.")
            return

        model_path = f"{ticker_dir}/model_{ticker.replace('.', '_')}_multi_step.keras"
        scaler_path = f"{ticker_dir}/scaler_{ticker.replace('.', '_')}_multi_step.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error(f"❌ Model atau scaler tidak ditemukan untuk ticker {ticker}. Silakan lakukan training terlebih dahulu.")
            return

        # Memuat model dan scaler
        status_text.text("🔄 Memuat model dan scaler...")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Memuat data saham dari Yahoo Finance
        status_text.text("🔄 Memuat data saham...")
        data = load_stock_data(ticker, start_date, end_date)

        # Menentukan fitur dan indeks dari kolom 'Close'
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        close_idx = features.index('Close')

        # Menyusun data input dan melakukan prediksi
        status_text.text("🔄 Menyusun data untuk prediksi...")
        predicted_prices, last_date = predict_future(model, scaler, data, time_step, n_steps, features, close_idx)

        # Validasi: pastikan ada cukup data untuk prediksi
        if predicted_prices is None:
            st.warning("❗ Tidak cukup data untuk prediksi.")
            return

        status_text.text("🔄 Menyelesaikan prediksi...")

        # Menyusun tanggal hasil prediksi ke depan
        predicted_dates = [last_date + timedelta(days=i + 1) for i in range(n_steps)]

        # Menampilkan hasil prediksi harga
        st.write("📈 Prediksi Harga:")
        for i in range(n_steps):
            st.write(f"{predicted_dates[i].strftime('%Y-%m-%d')}: Rp {predicted_prices[0, i]:,.2f}")

        # Membuat grafik harga historis dan hasil prediksi
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data.index[-60:], data['Close'][-60:], label='Harga Historis', color='blue')
        for i in range(n_steps):
            ax.scatter(predicted_dates[i], predicted_prices[0, i], color='red', label='Prediksi' if i == 0 else "")
        ax.set_title(f'Prediksi {n_steps} Hari ke Depan Saham {ticker}')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga (Rp)')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        st.pyplot(fig)

        # Selesaikan progress bar
        progress_bar.progress(100)
