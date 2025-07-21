#Import the libraries:
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date , timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import yfinance as yf     
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

#Title:
app_name = "Stock Market Forecasting App" 
st.title(app_name)
st.subheader("This App is created for Forecasting purpose of Stock Market dataset of a company")
st.image("https://i.pinimg.com/736x/de/2e/30/de2e30ae755cfee42e24935cddad5c8f.jpg")

#Siderbar:
st.sidebar.header("Navigation")

start_date = st.sidebar.date_input('Start date', date(2023,5,5))
end_date = st.sidebar.date_input("Last_date", date(2024,5,5))

#Add the ticker Symbol list:
ticker_list = ["AAPL", "GOOG", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker_symbol = st.sidebar.selectbox("Select a Stock", ticker_list)

#Now , we can fetch the dataset from the the user input with yfinance:
data = yf.download(ticker_symbol, start=start_date, end=end_date)
#Add as a column to the dataframe:
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write("Data from", start_date, "to", end_date)
st.write(data)
#Plot the dataset:
st.header("Data Visualization")
st.subheader("Plot of data")
st.write("**Note:** Select your specific data Range from the sidebar or an zoom on the plot, on your own choice!")
fig = px.line(data, x="Date", y=data.columns, title="Closing price of Stock Market", width=1000, height=600)
st.plotly_chart(fig, use_container_width=True)

#Create a column:
column = st.selectbox("Select the column to be used for forecasting", data.columns[1:])

#Add a selection box to choice a column for forecasting:
data = data[["Date", column]]
st.write("Selected Data")
st.write(data)

#ADF test to check the stationarity of dataset:
st.header("Is data Stationary")
st.write(adfuller(data[column])[1] < 0.05)
#Data Decomposition:
st.header("Decomposition of Data")
decompose = seasonal_decompose(data[column], model="additive", period=12)
st.write(decompose.plot())
#Make the same plot in plotly:
st.write("## Plotting the Decomposition in the Plotly")
st.plotly_chart(px.line(x=data["Date"], y=decompose.trend, title="Trend", width=1000, height=400, labels="Trend of Stock Market data"))
st.plotly_chart(px.line(x=data["Date"], y=decompose.seasonal, title="Seasonility", width=1000, height=400, labels={'x':"Date", "y":"Price"}).update_traces(line_color="Green"))
st.plotly_chart(px.line(x=data["Date"], y=decompose.resid, title="Residuals", width=1000, height=400, 
                        labels={'x':"Date", "y":"Price"}).update_traces(line_color="Red", line_dash="dot"))
#Model Selection:
models = ['SARIMA', 'Random Forest', 'Prophet', 'LSTM']
model_selection = st.sidebar.selectbox("Select the Model Forecasting", models)
#SARIMA Model Implementation:
if model_selection == 'SARIMA':
    p = st.slider("Select the value of p:", 0, 5, 2)
    d = st.slider("Select the value of d:", 0, 5, 1)
    q = st.slider("Select the value of q:", 0, 5, 2)
    seasonal_order = st.number_input("Select the values of seasonal p", 0, 24, 12)
    model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model_fit = model.fit()
    #Print the model summary:
    st.header("Model Summary")
    st.write(model_fit.summary())
    st.write("----")
    #Forecasting with SARIMA model:
    st.write("<p style='color:green; font-size= 50px; font-width=bold;'>Forecasting the Data with SARIMA</p>",
             unsafe_allow_html=True)
    forecast_period = st.number_input("Select the number of days forecast", 1, 365, 10)
    #Predict the future values:
    predictions = model_fit.get_prediction(start=len(data), end=len(data) + forecast_period)
    predictions = predictions.predicted_mean
    #Add the Index to the predictions:
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, "Date", predictions.index, True)
    predictions.reset_index(drop=True, inplace=True)
    st.write("Prediction", predictions)
    st.write("Actual data", data)
    st.write("---")
    #Plot the forecasted data with actual data:
    fig = go.Figure()
    # Add actual data to the plot
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    # Add predicted data to the plot
    fig.add_trace(
        go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted',
                   line=dict(color='red')))
    # Set the title and axis labels
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    # Display the plot
    st.plotly_chart(fig)

elif model_selection == 'Random Forest':
    # Random Forest Model:
    st.header("Random Forest Regressor")
    #Splitting the Training and Testing data:
    train_size = int(len(data)*0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    #Feature Engineering:
    train_X, train_y = train_data['Date'], train_data[column]
    test_X, test_y = test_data['Date'], test_data[column]

    #Inatialize and fit the random forest model:
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(train_X.values.reshape(-1, 1), train_y.values)
    #Predict the feature values:
    predictions = model_rf.predict(test_X.values.reshape(-1, 1))
    #Calculate the mean square error:
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)
    st.write("Root Mean Squared Error:", rmse)
    #Combine the training and testing data form plotting:
    combined_data = pd.concat([train_data, test_data])
    #Plot the actual and predicted values:
    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[column], mode='lines', name='Actual',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data["Date"], y=predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (Random Forest)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
    st.plotly_chart(fig)
elif model_selection == 'LSTM':
  # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

# Splitting the Training and Testing data
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Create a sequence for the LSTM model
    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length, 0])
            y.append(dataset[i + seq_length, 0])
        return np.array(X), np.array(y)


    seq_length = st.slider('Select the sequence length', 1, 30, 10)

    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)
    #Reshape the data for the LSTM model:
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
    #Inatialize and fit the LSTM model:
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(train_X, train_y, epochs=10, batch_size=16)
    #Predict the feature values:
    train_predictions = lstm_model.predict(train_X)
    test_predictions = lstm_model.predict(test_X)
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    #Evaluate the model:
    # Calculate mean squared error
    train_mse = mean_squared_error(train_data[seq_length:], train_predictions)
    train_rmse = np.sqrt(train_mse)
    test_mse = mean_squared_error(test_data[seq_length:], test_predictions)
    test_rmse = np.sqrt(test_mse)

    st.write(f"Train RMSE: {train_rmse}")
    st.write(f"Test RMSE: {test_rmse}")
    #Combine the train and test dataset for plotting:
    # Combine training and testing data for plotting
    train_dates = data['Date'][:train_size + seq_length]
    test_dates = data['Date'][train_size + seq_length:]
    combined_dates = pd.concat([train_dates, test_dates])
    combined_predictions = np.concatenate([train_predictions, test_predictions])
    #Plot the Actual vs Predicted values:
      #Plot the actual and predicted values:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_dates, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=combined_predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (LSTM)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
    st.plotly_chart(fig)
elif model_selection == 'Prophet':
    #Prophet Model:
    st.header("Prophet Facebook Model")
    #Prepare the Model:
    prophet_data = data[['Date', column]]
    prophet_data = prophet_data.rename(columns={'Date': 'ds', column: 'y'})
    #Create the Prophet Model:
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    #Forecast the feature values:
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)
    #Plot the model:
    fig = prophet_model.plot(forecast)
    plt.title("Forecast with Facebook Prophet")
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(fig)
st.write("Model Selected:", model_selection)




    
    
    




