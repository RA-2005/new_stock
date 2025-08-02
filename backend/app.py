import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from model import train_predict_model
import requests
from io import StringIO

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# 🔑 Replace with your own Alpha Vantage API key
ALPHA_VANTAGE_KEY = "GNX2N826PMRPDFZK"
def fetch_alpha_vantage_crypto(symbol):
    crypto, market = symbol[:3], symbol[3:]

    url = (
        f"https://www.alphavantage.co/quxery"
        f"?function=DIGITAL_CURRENCY_DAILY"
        f"&symbol={crypto}"
        f"&market={market}"
        f"&apikey={ALPHA_VANTAGE_KEY}"
        f"&datatype=csv"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to fetch from Alpha Vantage.")

    if response.text.startswith("Note") or response.text.startswith("Error"):
        raise Exception(f"Alpha Vantage message: {response.text.splitlines()[0]}")

    df = pd.read_csv(StringIO(response.text))
    print("Fetched crypto data preview:")
    print(df.head())
    print("Number of rows fetched:", df.shape[0])
    print("Columns:", df.columns)

    if df.empty:
        raise Exception("Alpha Vantage returned an empty CSV.")

    # Look for lowercase 'close' column
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    if not close_col:
        raise Exception("No 'close' column found in data.")

    df = df[['timestamp', close_col]].rename(columns={'timestamp': 'Date', close_col: 'Close'})
    df = df[::-1]  # chronological order
    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        coin = request.form['coin']  # e.g. BTCUSD

        try:
            df = fetch_alpha_vantage_crypto(coin)
            predicted = train_predict_model(df)
        except Exception as e:
            return f"Error fetching or processing data: {e}"

        days = list(range(1, 8))
        plt.figure(figsize=(8, 5))
        plt.plot(days, predicted, marker='o')
        plt.title(f'{coin} - Next 7 Days Prediction')
        plt.xlabel('Day')
        plt.ylabel('Price (USD)')
        plt.grid(True)

        chart_path = os.path.join('static', 'chart.png')
        plt.savefig(chart_path)
        plt.close()

        return render_template('result.html',
                               coin=coin,
                               img_path=chart_path,
                               predicted=predicted.flatten())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
