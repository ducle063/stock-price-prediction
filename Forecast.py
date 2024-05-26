# forecast_model.py
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def forecast_stock(code, period="2y", interval='1d'):
    df = yf.download(code, period=period, interval=interval)
    df = df.asfreq('D').ffill()

    ts = df[['Close']]

    cap = int(ts.size * .9)
    ts_train = ts.iloc[:cap]
    ts_test = ts.iloc[cap:]

    predictions = []
    actual_labels = []
    train_series = list(ts_train.Close)
    test_series = list(ts_test.Close)

    for i in range(len(test_series)):
        model = ARIMA(train_series, order=(0, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        predictions.append(forecast)
        actual_label = 1 if test_series[i] > train_series[-1] else 0
        actual_labels.append(actual_label)
        train_series.append(test_series[i])

    predicted_labels = [1 if predictions[i] > train_series[len(ts_train) + i - 1] else 0 for i in range(len(predictions))]

    # Đánh giá mô hình
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels)
    recall = recall_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels)
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conf_matrix": conf_matrix
    }
