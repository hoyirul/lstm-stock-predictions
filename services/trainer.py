import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from sklearn.preprocessing import MinMaxScaler
from services.utils import create_dataset

def train_model(data, time_step, n_steps, epochs, batch_size, features, close_idx, model_path, scaler_path, st, stop_flag):
    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    # Membagi data menjadi data latih dan data uji
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - time_step:]  # Tetap sertakan beberapa data sebelumnya

    # Membuat dataset untuk pelatihan dan pengujian
    X_train, y_train = create_dataset(train_data, time_step, n_steps, close_idx)
    X_test, y_test = create_dataset(test_data, time_step, n_steps, close_idx)

    # Membentuk ulang data input agar sesuai dengan format input LSTM
    X_train = X_train.reshape(X_train.shape[0], time_step, len(features))
    X_test = X_test.reshape(X_test.shape[0], time_step, len(features))

    # Membangun arsitektur model LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_step, len(features))),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32),
        Dense(n_steps)
    ])

    # Kompilasi model dengan optimizer dan fungsi loss
    model.compile(optimizer=Nadam(), loss='mean_squared_error')

    # Komponen UI untuk Streamlit: progress bar dan status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    epoch_logs = []

    # Callback untuk update setiap akhir epoch
    def on_epoch_end(epoch, logs):
        if stop_flag():  # Mengecek apakah user ingin menghentikan training
            model.stop_training = True
            status_text.text("⛔ Training dihentikan oleh pengguna.")
            return
        # Update progress bar dan status training
        percent_complete = int((epoch + 1) / epochs * 100)
        progress_bar.progress(min(percent_complete, 100))
        status_text.text(f"⏳ Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.6f}")
        epoch_logs.append((epoch, logs['loss']))

    # Early stopping untuk menghentikan training jika tidak ada peningkatan
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Melatih model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,  # Supaya tidak print log di terminal
        callbacks=[early_stop, LambdaCallback(on_epoch_end=on_epoch_end)]
    )

    # Menyimpan model dan scaler ke file
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Mengembalikan model, scaler, dan data uji untuk evaluasi selanjutnya
    return model, scaler, X_test, y_test
