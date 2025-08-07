import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import load_model

# LOAD MY MODEL
model=load_model('my_model.keras')

st.title('STOCK TREND PREDICTION')

ticker=st.text_input('Enter Stock Ticker','BAJFINANCE.NS')
df=yf.download(ticker,start='2010-01-01',end='2024-12-31', auto_adjust=False)

# --- Fetch Data ---
if ticker:
    df = yf.download(ticker, start='2010-01-01', end='2024-12-31')

    if df.empty:
        st.error(f"No data found for ticker: {ticker}")
        st.stop()

    st.subheader(f"Showing data for: {ticker}")
    st.write(df.tail())

# Describing data
st.subheader('Data From 2010-2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price Vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel("Time Scale")
plt.ylabel("Closing Price")
st.pyplot(fig)

ma100= df.Close.rolling(100).mean()
st.subheader('Closing Price Vs Time Chart with 100MA')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'g', label='Closing Price')
plt.plot(ma100, 'r', label='MA 100')
plt.xlabel("Time Scale")
plt.ylabel("Closing Price")
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b',label='Closing Price')
plt.plot(ma100,'r',label='Moving Avg 100')
plt.plot(ma200,'g', label='Moving Avg 200')
plt.xlabel("Time Scale")
plt.ylabel("Closing Price")
plt.legend()
st.pyplot(fig)

# SPLITTING THE DATA INTO TRAINING & TESTING
data=int(len(df)*.7)
data_training=pd.DataFrame(df['Close'][0:int(len(df)*.7)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
#scaled_training_data=scaler.fit_transform(data_training)

#  SPLITTING THE DATA INTO x_train & y_train
#x_train=[]
#y_train=[]
#for i in range(100, scaled_training_data.shape[0]): # (100,1722)
   # x_train.append(scaled_training_data[i-100:i])
   # y_train.append(scaled_training_data[i,0])

#x_train, y_train=np.array(x_train), np.array(y_train)

# TESTING PART
#past_100_days=data_training.tail(100)
#final_df=pd.concat([past_100_days,data_testing], ignore_index=True)
input_data=scaler.fit_transform(data_testing)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]): # (100 ,839)
     x_test.append(input_data[i-100:i]) # 0 : 100
     y_test.append(input_data[i,0]) # 100, 0th co

x_test, y_test=np.array(x_test), np.array(y_test)

y_predicted=model.predict(x_test)

test_dates = data_testing.index[100:]

# ORIGINAL PRICE vs PREDICTED PRICE
scaler=scaler.scale_
scaler_factor=1/scaler[0]
y_predicted=y_predicted*scaler_factor
y_test=y_test*scaler_factor

# Describing data
#st.subheader('ORIGINAL PRICE vs PREDICTED PRICE')
#st.write(y_predicted, y_test)

plot_index = df.index[data+100 : data+100 + len(y_test)]

plotting_data=pd.DataFrame({'Original_price': y_test.reshape(-1),
                            'predicted':y_predicted.reshape(-1)},
                            index=df.index[data+100:]
)

st.subheader('Predicted Price Vs Original Price')
fig=plt.figure(figsize=(12,6))
plt.plot(pd.concat([df.Close[:data+100],plotting_data],axis=0))
plt.xlabel("Time Scale")
plt.ylabel("Closing Price")
plt.legend(["Data - not used", "Predicted Closing Price", "Original Price"])
plt.show()
st.pyplot(fig)