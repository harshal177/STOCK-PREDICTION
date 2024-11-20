import streamlit as st
import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score

# Loading pretrained
model = load_model('keras_model.h5')


START = "2010-01-01"
TODAY = dt.date.today().strftime("%Y-%m-%d")


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

#  streamlit
st.title("Stock Price Prediction with LSTM")
st.write("This app uses an LSTM model to predict stock prices based on historical data.")


ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS):", "TCS.NS")

if ticker:
    
    data = load_data(ticker)
    st.subheader(f"Showing data for {ticker}")
    st.write(data.tail())

    
    # Preprocessing
    df = data.drop(['Date', 'Adj Close'], axis=1)
    train = pd.DataFrame(data[0:int(len(data)*0.70)])
    test = pd.DataFrame(data[int(len(data)*0.70): int(len(data))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = train.iloc[:, 4:5].values
    test_close = test.iloc[:, 4:5].values
    data_training_array = scaler.fit_transform(train_close)

    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100: i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Making predictions
    past_100_days = pd.DataFrame(train_close[-100:])
    test_df = pd.DataFrame(test_close)
    final_df = pd.concat([past_100_days, test_df], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    #  trained model
    y_pred = model.predict(x_test)

    # Rescale predictions and test values back to the original scale
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

   

    # Calculate Mean Absolute Error and R2 Score
    mae = mean_absolute_error(y_test, y_pred)
    mae_percentage = (mae / np.mean(y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    st.subheader(f"Model Evaluation")
    st.write(f"Mean Absolute Error: {mae_percentage:.2f}%")
    st.write(f"R2 Score: {r2:.2f}")

   

    # Predicting
    st.subheader("Next 7 Days Closing Price Prediction")
    
    # Start with the last 100 days from the test data
    last_100_days = input_data[-100:]
    predicted_prices = []

    for _ in range(7):
        
        input_seq = np.expand_dims(last_100_days, axis=0)
        next_pred = model.predict(input_seq)[0][0]

        # append predict value then updating sequence
        predicted_prices.append(next_pred)
        last_100_days = np.append(last_100_days[1:], [[next_pred]], axis=0)
    
    # Rescale the predictions to the original scale
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()
    
    # Displaying prediction
    future_dates = pd.date_range(data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7).strftime("%Y-%m-%d")
    next_7_days_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices})
    st.write(next_7_days_df)

    # Plot 
    st.subheader("Next 7 Days Predicted Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(future_dates, predicted_prices, marker='o', linestyle='-', color='orange')
    ax.set_title(f"Next 7 Days Stock Price Prediction for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Price (INR)")
    ax.grid(True)
    st.pyplot(fig)

