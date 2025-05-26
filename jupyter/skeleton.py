import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
import joblib
from datetime import datetime, timedelta
import time
import os

models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

tickers = [
    {
        "name": "Bank Negara Indonesia (BBNI)",
        "ticker": "BBNI.JK"
    }, 
    {
        "name": "Bank Rakyat Indonesia (BBRI)",
        "ticker": "BBRI.JK"
    },
    {
        "name": "Bank Tabungan Negara (BBTN)",
        "ticker": "BBTN.JK"
    },
    {
        "name": "Bank Mandiri (BMRI)",
        "ticker": "BMRI.JK"
    },
    {
        "name": "Bank Central Asia (BBCA)",
        "ticker": "BBCA.JK"
    },
    {
        "name": "Bank Negara Indonesia Syariah (BRIS)",
        "ticker": "BRIS.JK"
    },
    {
        "name": "Bank Syariah Indonesia (BSI)",
        "ticker": "BSI.JK"
    },
    {
        "name": "Bank Jago (ARTO)",
        "ticker": "ARTO.JK"
    },
    {
        "name": "Bank Danamon (BDMN)",
        "ticker": "BDMN.JK"
    },
    {
        "name": "Bank CIMB Niaga (BNGA)",
        "ticker": "BNGA.JK"
    }
]

# === Function Utilities ===
def create_dataset(dataset, time_step, n_steps, close_index):
    X, y = [], []
    for i in range(len(dataset) - time_step - n_steps + 1):
        X.append(dataset[i:i + time_step])
        y.append(dataset[i + time_step:i + time_step + n_steps, close_index])
    return np.array(X), np.array(y)

# === Streamlit Interface ===
st.title("📈 Aplikasi Prediksi Saham (LSTM Multi-Step)")

mode = st.sidebar.selectbox("Pilih Mode", ["Training", "Prediction"])
# Default ticker = "BBTN.JK"
ticker = st.sidebar.selectbox("Pilih Ticker", [t["ticker"] for t in tickers], format_func=lambda x: next(t["name"] for t in tickers if t["ticker"] == x))

time_step = st.sidebar.number_input("Time Step", min_value=1, value=60)
n_steps = st.sidebar.number_input("N Steps (Prediksi ke depan)", min_value=1, value=5)

# sumber
st.sidebar.write("Sumber Data: [Yahoo Finance](https://finance.yahoo.com/)")

if mode == "Training":
    st.header("🔧 Training Model")

    start_date = st.date_input("Start Date", value=datetime(2016, 1, 1))
    end_date = st.date_input("End Date", value=datetime(2024, 1, 1))
    epochs = st.number_input("Epochs", min_value=1, value=100)
    batch_size = st.number_input("Batch Size", min_value=1, value=16)
    verbose = st.selectbox("Verbose", [0, 1, 2], index=1)

    if st.button("🚀 Mulai Training"):
        stop_training_flag = st.session_state.get("stop_training", False)
        if st.button("🛑 Stop Training"):
            st.session_state["stop_training"] = True
            
        start_time = time.time()
        data = yf.download(ticker, start=start_date, end=end_date)
        data.dropna(inplace=True)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features])

        close_idx = features.index('Close')
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - time_step:]

        X_train, y_train = create_dataset(train_data, time_step, n_steps, close_idx)
        X_test, y_test = create_dataset(test_data, time_step, n_steps, close_idx)

        X_train = X_train.reshape(X_train.shape[0], time_step, len(features))
        X_test = X_test.reshape(X_test.shape[0], time_step, len(features))

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(time_step, len(features))),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32),
            Dense(n_steps)
        ])
        model.compile(optimizer=Nadam(), loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Custom epoch tracker
        epoch_logs = []

        def on_epoch_end(epoch, logs):
            if st.session_state.get("stop_training", False):
                model.stop_training = True
                status_text.text("⛔ Training dihentikan oleh pengguna.")
                return
            percent_complete = int((epoch + 1) / epochs * 100)
            progress_bar.progress(min(percent_complete, 100))
            status_text.text(f"⏳ Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.6f}")
            epoch_logs.append((epoch, logs['loss']))

        log_callback = LambdaCallback(on_epoch_end=on_epoch_end)

        status_text.text("🚀 Mulai training model...")
        st.session_state["stop_training"] = False
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop, log_callback]
        )
        status_text.text("✅ Training selesai!")
        progress_bar.progress(100)

        model_name = f"{models_dir}/model_{ticker.replace('.', '_')}_multi_step.keras"
        scaler_name = f"{models_dir}/scaler_{ticker.replace('.', '_')}_multi_step.pkl"
        model.save(model_name)
        joblib.dump(scaler, scaler_name)

        predicted_scaled = model.predict(X_test)
        pad_pred = np.zeros((len(predicted_scaled), len(features)))
        pad_pred[:, close_idx] = predicted_scaled[:, 0]
        predicted = scaler.inverse_transform(pad_pred)[:, close_idx]

        pad_actual = np.zeros((len(y_test), len(features)))
        pad_actual[:, close_idx] = y_test[:, 0]
        actual = scaler.inverse_transform(pad_actual)[:, close_idx]

        mape = mean_absolute_percentage_error(actual, predicted) * 100
        st.success(f"✅ Model dan scaler disimpan sebagai `{model_name}` & `{scaler_name}`")
        st.write(f"📉 MAPE pada data uji: {mape:.2f}%")

        # Plot
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

        st.write(f"⏱️ Waktu eksekusi: {time.time() - start_time:.2f} detik")

elif mode == "Prediction":
    st.header("🔮 Prediksi Harga Saham")

    if st.button("📤 Load Model dan Prediksi"):
        model_path = f"model_{ticker.replace('.', '_')}_multi_step.keras"
        scaler_path = f"scaler_{ticker.replace('.', '_')}_multi_step.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("❌ Model atau scaler tidak ditemukan. Silakan lakukan training terlebih dahulu.")
        else:
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            data = yf.download(ticker, start='2024-01-01', end=datetime.today().strftime('%Y-%m-%d'))
            data.dropna(inplace=True)
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            scaled_data = scaler.transform(data[features])
            close_idx = features.index('Close')

            X_input, _ = create_dataset(scaled_data, time_step, n_steps, close_idx)
            if len(X_input) == 0:
                st.warning("❗ Tidak cukup data untuk prediksi. Tambah periode waktu.")
            else:
                X_input = X_input.reshape(X_input.shape[0], time_step, len(features))
                predicted_scaled = model.predict(X_input)

                predicted_prices = []
                for i in range(predicted_scaled.shape[0]):
                    pad_pred = np.zeros((n_steps, len(features)))
                    pad_pred[:, close_idx] = predicted_scaled[i, :]
                    predicted_prices.append(scaler.inverse_transform(pad_pred)[:, close_idx])
                predicted_prices = np.array(predicted_prices)

                last_date = data.index[-1]
                predicted_dates = [last_date + timedelta(days=i+1) for i in range(n_steps)]

                st.write("📈 Prediksi Harga:")
                for i in range(n_steps):
                    st.write(f"{predicted_dates[i].strftime('%Y-%m-%d')}: Rp {predicted_prices[0, i]:,.2f}")

                # Plot
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
