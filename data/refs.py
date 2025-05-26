import yfinance as yf

def load_stock_data(ticker, start_date, end_date):
    """
    Mengambil data historis saham dari Yahoo Finance dan menghapus missing values.

    Args:
        ticker (str): Kode saham (misalnya 'BBRI.JK').
        start_date (str or datetime): Tanggal awal pengambilan data.
        end_date (str or datetime): Tanggal akhir pengambilan data.

    Returns:
        DataFrame: Data harga saham yang sudah dibersihkan.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data.dropna()
