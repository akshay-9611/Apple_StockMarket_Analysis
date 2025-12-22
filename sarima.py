import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import timedelta

st.set_page_config(page_title="30-Day Stock Predictor (SARIMAX)", layout="wide")
st.title("📈 Stock Price Predictor – Historical View + 30-Day Forecast")
CSV_PATH = "data/AAPL.csv"
MODEL_PATH = "sarimax_model.pkl"
SCALER_PATH = "feature_scaler.pkl"


df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date')
df = df.set_index('Date')
df = df.asfreq('B')
df = df.ffill()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

st.subheader("📅 Select Date Range to View Historical Trends")

min_date = df.index.min().date()
max_date = df.index.max().date()

start_date, end_date = st.date_input(
    "Choose date range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(label="Start date", min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input(label="End date", min_value=min_date, max_value=max_date)

filtered_df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]


st.subheader("📊 Historical Closing Price (Selected Range)")
st.line_chart(filtered_df["Close"])

st.write(filtered_df.tail())


spx = yf.download("^GSPC", start=df.index.min(), end=df.index.max())
spx["SPX_Close"] = spx["Close"]
df["SPX_Close"] = spx["SPX_Close"]
df = df.ffill()
for lag in [1, 2, 3, 5, 7]:
    df[f"Close_lag{lag}"] = df["Close"].shift(lag)


df["SPX_Return"] = df["SPX_Close"].pct_change()
df["SPX_Trend_5"] = df["SPX_Close"].rolling(5).mean()

df["SPX_Close_lag1"] = df["SPX_Close"].shift(1)
df["SPX_Return_lag1"] = df["SPX_Return"].shift(1)
df["SPX_Trend_5_lag1"] = df["SPX_Trend_5"].shift(1)

df["Rolling_Mean_20_lag1"] = df["Close"].rolling(20).mean().shift(1)
df["Rolling_Std_20_lag1"] = df["Close"].rolling(20).std().shift(1)

df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["MA_50"] = df["Close"].rolling(50).mean()

df["Volatility_10"] = df["Close"].rolling(10).std()
df["Volatility_20"] = df["Close"].rolling(20).std()

df["Daily_Return"] = df["Close"].pct_change()
df["Daily_Return_lag1"] = df["Daily_Return"].shift(1)


df["Log_Volume"] = np.log1p(df["Volume"])

df = df.dropna()

exog_cols = [
    'SPX_Close_lag1','Daily_Return_lag1','SPX_Return_lag1','SPX_Trend_5_lag1',
    'Rolling_Mean_20_lag1','Rolling_Std_20_lag1','SPX_Close','SPX_Return',
    'SPX_Trend_5','Close_lag1','Close_lag2','Close_lag3','Close_lag5','Close_lag7',
    'Open','High','Low','MA_10','MA_20','Daily_Return','Volatility_10',
    'Volatility_20','Log_Volume'
]

exog_full = df[exog_cols].copy()

trained_features = scaler.feature_names_in_

for col in trained_features:
    if col not in exog_full.columns:
        exog_full[col] = 0.0

exog_full = exog_full[trained_features]

exog_scaled_full = scaler.transform(exog_full)
exog_scaled_full = pd.DataFrame(exog_scaled_full, index=df.index, columns=trained_features)

sarimax_cols = ['Low','High','Open','SPX_Return','Close_lag1','SPX_Trend_5_lag1']
exog_sarimax = exog_scaled_full[sarimax_cols]

if st.button("🚀 Predict 30 Days After Selected Range"):

    prediction_start_date = pd.to_datetime(end_date)

   
    nearest_index = df.index.asof(prediction_start_date)
    last_exog = exog_sarimax.loc[nearest_index].values.reshape(1, -1)
    future_exog = np.repeat(last_exog, 30, axis=0)

    # Future dates
    future_dates = pd.date_range(
        start=prediction_start_date + timedelta(days=1),
        periods=30,
        freq='B'
    )

    # Forecast
    forecast = model.get_forecast(steps=30, exog=future_exog)
    predictions = forecast.predicted_mean

    result_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": predictions
    })

    st.subheader("📌 30-Day Stock Price Forecast")
    st.dataframe(result_df)

    st.subheader("📈 Forecast Trend")
    st.line_chart(result_df.set_index("Date"))

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=result_df.to_csv(index=False).encode(),
        file_name="30_day_forecast.csv"
    )
