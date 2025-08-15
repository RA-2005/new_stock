import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from model import train_predict_model
import requests
import uuid  # for unique chart names

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# 🔑 Replace with your own Alpha Vantage API key
ALPHA_VANTAGE_KEY = "GNX2N826PMRPDFZK"

def fetch_alpha_vantage_crypto(symbol):
    # Example: BTCUSD -> BTC + USD
    if len(symbol) < 6:
        raise ValueError("Invalid symbol format. Example: BTCUSD")

    crypto = symbol[:-3]
    market = symbol[-3:]

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=DIGITAL_CURRENCY_DAILY"
        f"&symbol={crypto}"
        f"&market={market}"
        f"&apikey={ALPHA_VANTAGE_KEY}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch from Alpha Vantage. HTTP status: {response.status_code}")

    data = response.json()

    if "Time Series (Digital Currency Daily)" not in data:
        raise Exception(f"Alpha Vantage error: {data.get('Note') or data.get('Error Message') or 'Unexpected response'}")

    time_series = data["Time Series (Digital Currency Daily)"]

    df = pd.DataFrame([
        {
            'Date': date,
            'Close': float(day_data.get('4b. close (USD)') or day_data.get('4a. close (USD)'))
        }
        for date, day_data in time_series.items()
        if day_data.get('4b. close (USD)') or day_data.get('4a. close (USD)')
    ])

    price_df = df.sort_values('Date')
    return price_df


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        coin = request.form['coin'].upper().strip()  # normalize input

        try:
            df = fetch_alpha_vantage_crypto(coin)
            predicted = train_predict_model(df)
            predicted_list = predicted.flatten().tolist() if hasattr(predicted, 'flatten') else list(predicted)
        except Exception as e:
            return render_template('error.html', error=str(e))

        # Create unique chart name to avoid browser caching
        chart_filename = f"chart_{uuid.uuid4().hex}.png"
        chart_path = os.path.join('static', chart_filename)

        days = list(range(1, len(predicted_list) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(days, predicted_list, marker='o')
        plt.title(f'{coin} - Next {len(days)} Days Prediction')
        plt.xlabel('Day')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.savefig(chart_path)
        plt.close()

        return render_template('result.html',
                               coin=coin,
                               img_path=chart_path,
                               predicted=predicted_list)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
