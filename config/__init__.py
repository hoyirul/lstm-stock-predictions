import os

# Direktori penyimpanan model
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Daftar saham (tickers) yang didukung
tickers = [
    {"name": "Bank Negara Indonesia (BBNI)", "ticker": "BBNI.JK"},
    {"name": "Bank Rakyat Indonesia (BBRI)", "ticker": "BBRI.JK"},
    {"name": "Bank Tabungan Negara (BBTN)", "ticker": "BBTN.JK"},
    {"name": "Bank Mandiri (BMRI)", "ticker": "BMRI.JK"},
    {"name": "Bank Central Asia (BBCA)", "ticker": "BBCA.JK"},
    {"name": "Bank Negara Indonesia Syariah (BRIS)", "ticker": "BRIS.JK"},
    {"name": "Bank Syariah Indonesia (BSI)", "ticker": "BSI.JK"},
    {"name": "Bank Jago (ARTO)", "ticker": "ARTO.JK"},
    {"name": "Bank Danamon (BDMN)", "ticker": "BDMN.JK"},
    {"name": "Bank CIMB Niaga (BNGA)", "ticker": "BNGA.JK"},
]
