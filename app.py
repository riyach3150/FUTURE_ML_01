
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Sales Forecasting App")

st.write("This app predicts future monthly sales using a Regression model.")

# Sample demo dataset
dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
sales = np.random.randint(1000, 5000, 36)

df = pd.DataFrame({"Date": dates, "Sales": sales})

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

X = df[['Year', 'Month', 'Quarter']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

months = st.slider("Select number of months to forecast", 1, 12, 6)

future_dates = pd.date_range(start=df['Date'].max(), periods=months+1, freq='M')[1:]
future_df = pd.DataFrame({"Date": future_dates})

future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['Quarter'] = future_df['Date'].dt.quarter

predictions = model.predict(future_df[['Year','Month','Quarter']])

future_df['Forecasted Sales'] = predictions

st.subheader("📈 Forecasted Sales")
st.write(future_df)
