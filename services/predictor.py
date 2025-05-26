import numpy as np
from services.utils import create_dataset

def predict_future(model, scaler, data, time_step, n_steps, features, close_idx):
    # Melakukan transformasi data menggunakan scaler yang sudah dilatih sebelumnya
    scaled_data = scaler.transform(data[features])

    # Membuat dataset input untuk prediksi berdasarkan parameter time_step dan n_steps
    X_input, _ = create_dataset(scaled_data, time_step, n_steps, close_idx)

    # Jika data tidak cukup untuk membentuk input, kembalikan None
    if len(X_input) == 0:
        return None, None

    # Mengubah bentuk input agar sesuai dengan input shape model LSTM: (samples, time_step, fitur)
    X_input = X_input.reshape(X_input.shape[0], time_step, len(features))

    # Melakukan prediksi menggunakan model LSTM yang telah dilatih
    predicted_scaled = model.predict(X_input)

    predicted_prices = []

    # Proses inverse scaling agar hasil prediksi bisa dikembalikan ke skala harga asli
    for i in range(predicted_scaled.shape[0]):
        # Membuat array kosong dengan ukuran (n_steps, jumlah fitur) untuk melakukan inverse transform
        pad_pred = np.zeros((n_steps, len(features)))

        # Hanya mengisi kolom 'Close' dengan hasil prediksi
        pad_pred[:, close_idx] = predicted_scaled[i, :]

        # Melakukan inverse transform dan mengambil nilai kolom 'Close' saja
        predicted_prices.append(scaler.inverse_transform(pad_pred)[:, close_idx])

    # Mengembalikan array hasil prediksi dan tanggal terakhir dari data historis
    return np.array(predicted_prices), data.index[-1]
