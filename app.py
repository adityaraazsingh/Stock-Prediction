
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model

import streamlit as st
import matplotlib.pyplot as plt

from datetime import datetime

#C:\Users\adity\Documents\-PRIORITY-\Stock Prediction\Stock Model Predtion real .keras

# streamlit run "C:\Users\adity\Documents\-PRIORITY-\Stock Prediction\app.py"     
model = load_model("C:\\Users\\adity\\Documents\\-PRIORITY-\\Stock Prediction\\Stock Model Predtion real .keras")

st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2024-01-01'
start = st.date_input("Select start date", datetime.today())
end = st.date_input("Select end date", datetime.today())
   

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test],ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


st.subheader('Price vs MA-50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r', label = 'MA 50 Days')
plt.plot(data.Close,'g')
plt.legend() 
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA-50 vs MA-100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r', label = 'MA 50 Days')
plt.plot(ma_100_days,'b',label = 'MA 100 Days')
plt.plot(data.Close,'g')
plt.legend() 
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA-50 vs MA-100 vs MA-200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r',label = 'MA 100 Days')
plt.plot(ma_200_days,'b', label = 'MA 200 Days')
plt.plot(data.Close,'g')
plt.legend() 
plt.show()
st.pyplot(fig3)

x = []
y = []


for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale =1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict,'r',label = 'Predicted Price')
plt.plot(y,'g', label = 'Original price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
plt.legend() 
st.pyplot(fig4)

