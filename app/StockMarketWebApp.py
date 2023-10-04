# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Title
app_name = "Stock Market Forecasting App"
st.title(app_name)
st.subheader("This App is created to forecast the stock prices of selected company using ARIMA model")
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg?size=626&ext=jpg",width=700)

# take input from the user of app about the start and end date

# sidebar
st.sidebar.subheader("Select the parameters from below")

start_date = st.sidebar.date_input("Start Date",date(2020,1,1))
end_date = st.sidebar.date_input("End Date",date(2020,12,31))

# add ticker symbol list

ticker_list = ["AAPL", "MSFT", "GOOGL", "GODG", "META", "TSLA", "NVDA","ADBE", "PYPL","INTC","CMCSA","NFLX","PEP"]
ticker = st.sidebar.selectbox("Select the company",ticker_list)

# fetch data from user inputs using yfinance library

data = yf.download(ticker,start=start_date,end=end_date)

# add date as a column to the dataframe
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write("Data from",start_date,"to",end_date)
st.write(data)

# Lets plot the data
st.header("Data Visualization")
st.subheader("Time Series Plot")
fig = px.line(data,x="Date",y = data.columns,title="Time Series Plot",height=600,width=1000)
st.plotly_chart(fig)

# add a select box to select box to select column from data
column = st.selectbox("Select the column to be used for forecasting",data.columns[1:])

# subsetting the data
data = data[["Date",column]]
st.write("Selected Data")
st.write(data)

# ADF test for checking the stationarity of the data
st.header("Is header stationary?")
st.write(adfuller(data[column])[1]<0.05) 

# lets decompose the data
st.header("Decomposition of the data")
decomposition = sm.tsa.seasonal_decompose(data[column],model="additive",period=12)
st.write(decomposition.plot())

# make same plot in plotly
st.write("## PLotting the Decomposition plot in Plotly")
st.plotly_chart(px.line(x = data["Date"],y = decomposition.trend,title="Trend",height=600,width=1200,labels={"x":"Date","y":"Price"}).update_traces(line_color="red"))
st.plotly_chart(px.line(x = data["Date"],y = decomposition.seasonal,title="Seasonality",height=600,width=1200,labels={"x":"Date","y":"Price"}).update_traces(line_color="green"))
st.plotly_chart(px.line(x = data["Date"],y = decomposition.resid,title="Residual",height=600,width=1200,labels={"x":"Date","y":"Price"}).update_traces(line_color="blue",line_dash="dot"))

# Lets run the model
# user input for three parameters of the model and seasonal order
p = st.slider("Select the value of p",0,5,2)
d = st.slider("Select the value of d",0,5,1)
q = st.slider("Select the value of q",0,5,2)
seasonal_order = st.number_input("Select the value of seasonal p",0,24,12)

model = sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

# print Model Summary
st.header("Model Summary")
st.write(model.summary())
st.write("## Model Diagnostics")

# predict the future values (forecasting)
forecast_period = st.number_input("Enter the number of days for forecasting",1,365,30)

# predict the future values
predictions = model.get_prediction(start=len(data),end=len(data)+forecast_period)
predictions = predictions.predicted_mean

# add index to the predictions
predictions.index = pd.date_range(start=end_date,periods= len(predictions),freq="D")
predictions = pd.DataFrame(predictions)
predictions.insert(0,"Date",predictions.index,True)
st.write("Predictions",predictions)
st.write("Actual Data",data)
st.write("## Plotting the Actual and Predicted Values")

# lets plot the data
fig = go.Figure()
# add actual data to the plot
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode="lines+markers",name="Actual Values",line=dict(color="blue",width=2)))
# add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions["predicted_mean"],mode="lines+markers",name="Predicted Values",line=dict(color="red",width=2)))
# set the title and axis labels
fig.update_layout(title="Actual vs Predicted Values",xaxis_title="Date",yaxis_title="Price",height=600,width=1000)
# display the plot
st.plotly_chart(fig)

# Add buttons to show and hide separate plots
show_plots = False
if st.button("Show Separate Plots"):
    if not show_plots:
        st.write(px.line(x=data["Date"],y=data[column],title="Actual Values",height=600,width=1200,labels={"x":"Date","y":"Price"}).update_traces(line_color="blue"))
        st.write(px.line(x=predictions["Date"],y=predictions["predicted_mean"],title="Predicted Values",height=600,width=1200,labels={"x":"Date","y":"Price"}).update_traces(line_color="red"))
        show_plots = True
    else:
        show_plots = False
        # add hide plots button
        if st.button("Hide Separate Plots"):
            if not hide_plots:
                hide_plots = True
        else:
            hide_plots = False
            
st.write("--------------")
st.write("Created by: [Ammar Zafar](https://www.linkedin.com/in/ammar-zafar-7345a2156/)")

st.write("<p style = 'text-align:center'>Created by: <a href='https://www.linkedin.com/in/ammar-zafar-7345a2156/'>Ammar Zaffar</a></p>",unsafe_allow_html=True)

# paste youtube icon from online source with link

st.write("## Connect with me on Social Media")
# add links to my social media
# urls of the images
github_url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
LinkedIn_url = ("[Linkedin](https://www.linkedin.com/in/ammar-zafar-7345a2156/)")
twitter_url = "https://www.freepnglogos.com/uploads/twitter-logo-png/twitter-logo-vector-png-clipart-1.png"
facebook_url = "https://www.freepnglogos.com/uploads/facebook-logo-png/facebook-logo-png-facebook-vector-logo-png-clipart-1.png"

# redirect urls
github_redirect_url = "https://github.com/AmmarZafar028?tab=repositories"
LinkedIn_redirect_url = "https://www.linkedin.com/in/ammar-zafar-7345a2156/"
twitter_redirect_url =  "https://twitter.com/ammarzafar028"
facebook_redirect_url = "https://www.facebook.com/profile.php?id=100004547232843"

# add links to the images
st.markdown("<a href="+github_redirect_url+"><img src="+github_url+" width=50>  </a>",unsafe_allow_html=True)
st.markdown("<a href="+LinkedIn_redirect_url+"><img src="+LinkedIn_url+" width=50></a>",unsafe_allow_html=True)
st.markdown("<a href="+twitter_redirect_url+"><img src="+twitter_url+" width=50></a>",unsafe_allow_html=True)
st.markdown("<a href="+facebook_redirect_url+"><img src="+facebook_url+" width=50></a>",unsafe_allow_html=True)