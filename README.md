# LSTM Stock Predictions and Streamlit
> This project uses LSTM (Long Short-Term Memory) neural networks to predict stock prices and visualizes the results using Streamlit.

## Overview
This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices. The model is trained on historical stock data, and the predictions are visualized using Streamlit, a web application framework for Python.

## Requirements
- Python 3.6 or higher
- Libraries:
  - TensorFlow
  - Keras
  - NumPy
  - Pandas
  - Matplotlib
  - Streamlit
  - Scikit-learn
  - yfinance
  - Plotly
  - Seaborn
  - uv

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hoyirul/lstm-stock-predictions.git lstm-stock-predictions
   ```
2. Navigate to the project directory:
   ```bash
    cd lstm-stock-predictions
    ```
3. Install uv if not already installed (uv)[https://docs.astral.sh/uv/#tools]
4. Create a virtual environment:
    ```bash
    uv venv --python 3.11
    ```
3. Install the required libraries:
    ```bash
    uv pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
5. Open your web browser and go to `http://localhost:8501` to view the app.
