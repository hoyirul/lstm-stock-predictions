import numpy as np

def create_dataset(dataset, time_step, n_steps, close_index):
    """
    Membuat dataset untuk input dan target prediksi model LSTM multi-step.

    Args:
        dataset (ndarray): Data yang telah dinormalisasi (hasil dari scaler).
        time_step (int): Jumlah langkah waktu (historis) yang digunakan sebagai input.
        n_steps (int): Jumlah langkah ke depan yang ingin diprediksi.
        close_index (int): Indeks kolom 'Close' dalam dataset fitur.

    Returns:
        tuple: (X, y)
            - X: Array 3D untuk input model [samples, time_step, fitur]
            - y: Array 2D untuk target prediksi [samples, n_steps]
    """
    X, y = [], []

    # Iterasi untuk membuat sequence dari dataset
    for i in range(len(dataset) - time_step - n_steps + 1):
        # Mengambil window data sepanjang time_step untuk dijadikan input
        X.append(dataset[i:i + time_step])

        # Mengambil n langkah ke depan setelah time_step sebagai label y
        y.append(dataset[i + time_step:i + time_step + n_steps, close_index])

    # Mengembalikan hasil sebagai array numpy
    return np.array(X), np.array(y)
