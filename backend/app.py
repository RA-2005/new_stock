import os
from flask import Flask, render_template, request
import matplotlib.pyplot as plt 
from model import train_predict_model

app = Flask(__name__, template_folder="../templates", static_folder="../static")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        coin = request.form['coin']
        csv_path = f"backend/data/{coin}.csv"  # Use the selected coin name

        if not os.path.exists(csv_path):
            return f"Coin data for {coin} not found!"

        predicted = train_predict_model(csv_path)
        days = list(range(1, 8))

        # Plot and save prediction chart
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
