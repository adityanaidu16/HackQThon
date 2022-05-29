from flask import Flask, render_template, request
import csv
from qiskit_finance import QiskitFinanceError
from qiskit_finance.data_providers import *
import datetime
import pandas as pd
import numpy as np

# import python functions from helpers.py
from helpers import apology

from stocks import find_predicted_price

# Configure flask application
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# render index.html by default
@app.route('/')
def index():
    return render_template('index.html')

# function for searching stock symbols
@app.route("/", methods=["GET", "POST"])
def stocks():
    # User reach route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # request user input from HTML form
        stock_symbol = request.form.get("stock_response")

        try:
            data = YahooDataProvider(
                tickers=[stock_symbol],
                start=datetime.datetime(2018, 1, 1),
                end=datetime.datetime(2022, 5, 28)
            )
            data.run()
            df = pd.DataFrame(data._data)
            df = df.T

        except:
            return apology("Symbol not found", 400)

        predicted_price = find_predicted_price(stock_symbol)

        # render displayattractions.html and pass results
        return render_template('stocks.html', stock_symbol=stock_symbol, predicted_price=predicted_price)

    else:
        # render index.html until HTML form is submitted
        return render_template('index.html')

# render help.html when Help is clicked on nav-bar
@app.route("/help")
def help():
    return render_template('help.html')

if __name__ == "__main__":
    app.run(debug=True)