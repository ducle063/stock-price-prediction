# Web_demo.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from Forecast import forecast_stock

st.set_page_config(
    page_title="Dự đoán giá cổ phiếu",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.stAlert {
    font-size: 1.1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Ứng dụng dự đoán giá cổ phiếu")

st.sidebar.header("Lựa chọn cổ phiếu")
selected_stock = st.sidebar.text_input("Nhập mã cổ phiếu (ví dụ: AAPL, GOOG):", "AAPL")

st.sidebar.header("Cài đặt dự đoán")
num_prediction_days = st.sidebar.slider("Số ngày dự đoán:", min_value=1, max_value=30, value=5)

@st.cache_data
def load_data(stock, start, end):
    return yf.download(stock, start=start, end=end)

if selected_stock:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 20)  
    stock_data = load_data(selected_stock, start_date, end_date)

    if stock_data.empty:
        st.error(f"Không tìm thấy dữ liệu cho mã {selected_stock}. Vui lòng kiểm tra lại.")
    else:
        stock_data = stock_data.asfreq('D').ffill()
        ts = stock_data[['Close']]

        model = ARIMA(ts, order=(5, 1, 0))
        model_fit = model.fit()

        stock_data['Predicted_Close'] = model_fit.predict(start=0, end=len(ts)-1)

        last_date = stock_data.index[-1]
        next_n_days = pd.date_range(last_date + timedelta(days=1), periods=num_prediction_days, freq="B")

        predictions_n_days = []
        last_train_series = list(ts.Close)
        for _ in range(num_prediction_days):
            model = ARIMA(last_train_series, order=(5, 1, 0))
            model_fit = model.fit()
            next_day_prediction = model_fit.forecast(steps=1)[0]
            
            random_factor = np.random.normal(0, 1)  
            next_day_prediction += random_factor
            
            predictions_n_days.append(next_day_prediction)
            last_train_series.append(next_day_prediction)

        for i in range(len(next_n_days)):
            stock_data.loc[next_n_days[i], "Predicted_Close"] = predictions_n_days[i]
            stock_data.loc[next_n_days[i], "Close"] = predictions_n_days[i]
            for col in stock_data.columns:
                if col != "Predicted_Close" and col != "Close":
                    stock_data.loc[next_n_days[i], col] = np.nan 
                    
        st.subheader(f"Thông tin về {selected_stock}")
        stock_info = yf.Ticker(selected_stock).info
        st.write(f"**Tên công ty:** {stock_info['longName']}")
        st.write(f"**Ngành:** {stock_info['industry']}")
        st.write(f"**Vốn hóa:** {stock_info['marketCap']}")

        with st.expander("Xem dữ liệu"):
            st.dataframe(stock_data.round(2))

        st.subheader(f"Biểu đồ giá đóng cửa của {selected_stock} (bao gồm dự đoán {num_prediction_days} ngày)")

        ma_options = [50, 100, 200]
        for ma in ma_options:
            stock_data[f"MA_{ma}"] = stock_data["Close"].rolling(window=ma, min_periods=1).mean()

        fig = px.line(
            stock_data,
            x=stock_data.index,
            y=["Close", "Predicted_Close"] + [f"MA_{ma}" for ma in ma_options],
            title=f"Giá đóng cửa của {selected_stock} (dự đoán {num_prediction_days} ngày)",
        )
        fig.update_layout(
            autosize=True,
            height=600 
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Vui lòng nhập mã cổ phiếu.")
