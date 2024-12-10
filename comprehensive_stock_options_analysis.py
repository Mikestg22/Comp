
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date
from scipy.stats import norm
import requests

# Black-Scholes Options Pricing Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.error(f"No data found for ticker '{ticker}'.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Fetch options data for a stock
def fetch_options_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        st.sidebar.header("Select Options Expiry Date")
        selected_date = st.sidebar.selectbox("Expiry Date", options_dates)
        options_chain = stock.option_chain(selected_date)
        return options_chain
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return None

# Risk Management Recommendations
def calculate_risk(stock_data):
    st.subheader("Risk Analysis")
    volatility = stock_data['Close'].pct_change().std() * np.sqrt(252)
    st.write(f"Annualized Volatility: {volatility:.2%}")
    if volatility < 0.2:
        st.write("Risk Level: **Low**")
    elif 0.2 <= volatility <= 0.4:
        st.write("Risk Level: **Medium**")
    else:
        st.write("Risk Level: **High**")

# Predict Future Prices
def predict_prices(data, days=7):
    if len(data) < 2:
        st.error("Not enough data for price predictions.")
        return None
    data['Days'] = np.arange(len(data))
    X = data[['Days']].values
    y = data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

# Backtesting Strategy
def backtest_strategy(data, strategy_function):
    st.subheader("Backtesting Results")
    results = strategy_function(data)
    st.write("Results:", results)

# Sentiment Analysis from News
def fetch_news_sentiment(ticker):
    st.subheader("News Sentiment Analysis")
    # Placeholder for news sentiment logic (requires API like Alpha Vantage or NewsAPI)
    st.write("News sentiment analysis coming soon.")

# Streamlit App
st.title("Comprehensive Stock & Options Analysis App")

# Sidebar Inputs
st.sidebar.header("Input Options")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=date.today())
days_to_predict = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=7)

# Fetch stock data
if st.sidebar.button("Analyze Stock"):
    st.subheader(f"Analysis for {ticker}")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        st.write("Recent Stock Data")
        st.dataframe(stock_data.tail(10))

        # Risk Analysis
        calculate_risk(stock_data)

        # Price Predictions
        predictions = predict_prices(stock_data, days=days_to_predict)
        if predictions is not None:
            st.write(f"Predicted Prices for the next {days_to_predict} days:")
            st.write(predictions)

# Fetch options data
if st.sidebar.button("Analyze Options"):
    options_data = fetch_options_data(ticker)
    if options_data:
        st.subheader(f"Options Chain for {ticker}")
        st.write("Calls")
        st.dataframe(options_data.calls)
        st.write("Puts")
        st.dataframe(options_data.puts)

# News Sentiment Analysis
if st.sidebar.button("Fetch News Sentiment"):
    fetch_news_sentiment(ticker)
