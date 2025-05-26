import streamlit as st
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from services.trainer import train_model
from data.refs import load_stock_data
from sklearn.metrics import mean_absolute_percentage_error
import os
from config import models_dir

def show(ticker, time_step, n_steps):
    st.header("🔧 Training Model")

    # Input parameter dari user
    start_date = st.date_input("Start Date", value=datetime(2016, 1, 1))
    end_date = st.date_input("End Date", value=datetime(2024, 1, 1))
    epochs = st.number_input("Epochs", min_value=1, value=100)
    batch_size = st.number_input("Batch Size", min_value=1, value=16)

    # Ketika tombol "Mulai Training" diklik
    if st.button("🚀 Mulai Training"):
        # Ambil data saham
        data = load_stock_data(ticker, start_date, end_date)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        close_idx = features.index('Close')

        # Buat folder untuk menyimpan model jika belum ada
        ticker_dir = f"{models_dir}/{ticker.replace('.', '_')}"
        if not os.path.exists(ticker_dir):
            os.makedirs(ticker_dir)

        model_name = f"{ticker_dir}/model_{ticker.replace('.', '_')}_multi_step.keras"
        scaler_name = f"{ticker_dir}/scaler_{ticker.replace('.', '_')}_multi_step.pkl"

        # Fungsi untuk memeriksa apakah user ingin menghentikan training
        def stop_flag():
            return st.session_state.get("stop_training", False)

        # Tombol untuk menghentikan training (opsional)
        if st.button("🛑 Stop Training"):
            st.session_state["stop_training"] = True

        # Mulai timer untuk mengukur waktu training
        start_time = time.time()

        # Latih model menggunakan fungsi dari services.trainer
        model, scaler, X_test, y_test = train_model(
            data, time_step, n_steps, epochs, batch_size,
            features, close_idx, model_name, scaler_name, st, stop_flag
        )

        # Konfirmasi penyimpanan model
        st.success(f"Model dan scaler disimpan di `{model_name}` & `{scaler_name}`")

        # Prediksi terhadap data uji
        predicted_scaled = model.predict(X_test)

        # Transformasi hasil prediksi kembali ke skala asli
        import numpy as np
        pad_pred = np.zeros((len(predicted_scaled), len(features)))
        pad_pred[:, close_idx] = predicted_scaled[:, 0]
        predicted = scaler.inverse_transform(pad_pred)[:, close_idx]

        # Ambil nilai aktual (ground truth)
        pad_actual = np.zeros((len(y_test), len(features)))
        pad_actual[:, close_idx] = y_test[:, 0]
        actual = scaler.inverse_transform(pad_actual)[:, close_idx]

        # Hitung akurasi prediksi menggunakan MAPE
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        st.write(f"📉 MAPE pada data uji: {mape:.2f}%")

        # Visualisasi hasil prediksi vs aktual
        actual_dates = data.index[-len(actual):]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(actual_dates, actual, label='Harga Aktual')
        ax.plot(actual_dates, predicted, 'r--', label='Prediksi')
        ax.set_title(f'Prediksi Harga Saham {ticker}')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        st.pyplot(fig)

        # Tampilkan waktu yang dibutuhkan untuk proses training
        st.write(f"⏱️ Waktu eksekusi: {time.time() - start_time:.2f} detik")
