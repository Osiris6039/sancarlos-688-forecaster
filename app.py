# app.py
# AI Sales and Customer Forecast Web App

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
import os

# --- Authentication ---
def authenticate(username, password):
    return username == "admin" and password == "forecast123"

# --- Data Files ---
data_file = "data.csv"
event_file = "events.csv"

# --- Initial Setup ---
if not os.path.exists(data_file):
    pd.DataFrame(columns=["Date", "Sales", "Customers", "Weather", "AddOnSales"]).to_csv(data_file, index=False)

if not os.path.exists(event_file):
    pd.DataFrame(columns=["EventDate", "EventName", "LastYearSales", "LastYearCustomers"]).to_csv(event_file, index=False)

# --- Forecasting Function ---
def train_forecaster(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values("Date", inplace=True)
    df['Day'] = df['Date'].dt.dayofyear
    df['AddOnFlag'] = df['AddOnSales'].fillna(0).apply(lambda x: 1 if x > 0 else 0)
    features = ['Day', 'Weather', 'AddOnFlag']
    X = pd.get_dummies(df[features])
    sales_model = GradientBoostingRegressor().fit(X, df['Sales'])
    cust_model = GradientBoostingRegressor().fit(X, df['Customers'])
    return sales_model, cust_model, X.columns

def make_forecast(sales_model, cust_model, columns, df, events, days=10):
    today = datetime.today()
    forecast_dates = [today + timedelta(days=i) for i in range(days)]
    forecasts = []

    for d in forecast_dates:
        day = d.timetuple().tm_yday
        row = {"Day": day, "AddOnFlag": 0, "Weather": "Sunny"}
        event_boost = events[events["EventDate"] == d.strftime('%Y-%m-%d')]
        if not event_boost.empty:
            row["AddOnFlag"] = 1
        df_row = pd.DataFrame([row])
        df_row = pd.get_dummies(df_row).reindex(columns=columns, fill_value=0)
        sale = sales_model.predict(df_row)[0]
        cust = cust_model.predict(df_row)[0]
        if not event_boost.empty:
            sale += event_boost['LastYearSales'].values[0] * 0.15
            cust += event_boost['LastYearCustomers'].values[0] * 0.10
        forecasts.append((d.strftime('%Y-%m-%d'), round(sale), round(cust)))
    return forecasts

# --- Streamlit UI ---
st.set_page_config(page_title="AI Forecast App", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

st.title("📊 AI Sales & Customer Forecasting App")

# --- Load Data ---
data = pd.read_csv(data_file)
event_data = pd.read_csv(event_file)

# --- Daily Data Input ---
st.header("📥 Input Daily Data")
with st.form("daily_form", clear_on_submit=True):
    date = st.date_input("Date")
    sales = st.number_input("Sales", 0)
    customers = st.number_input("Customers", 0)
    weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])
    addon = st.number_input("Add-On Sales", 0)
    if st.form_submit_button("Submit Entry"):
        new_row = pd.DataFrame([{"Date": date, "Sales": sales, "Customers": customers, "Weather": weather, "AddOnSales": addon}])
        data = pd.concat([data, new_row], ignore_index=True)
        data["Date"] = pd.to_datetime(data["Date"])
        data.sort_values("Date", inplace=True)
        data.to_csv(data_file, index=False)
        st.success("Entry added!")

# --- Event Input ---
st.header("📅 Input Future Event")
with st.form("event_form", clear_on_submit=True):
    edate = st.date_input("Event Date")
    ename = st.text_input("Event Name")
    esales = st.number_input("Last Year's Sales", 0)
    ecustomers = st.number_input("Last Year's Customers", 0)
    if st.form_submit_button("Submit Event"):
        new_event = pd.DataFrame([{
            "EventDate": edate.strftime('%Y-%m-%d'),
            "EventName": ename,
            "LastYearSales": esales,
            "LastYearCustomers": ecustomers
        }])
        event_data = pd.concat([event_data, new_event], ignore_index=True)
        event_data.to_csv(event_file, index=False)
        st.success("Event added!")

# --- Show Data Records ---
st.subheader("📋 Daily Records")
for i in range(len(data)):
    st.write(data.iloc[i].to_dict())
    if st.button(f"Delete {data.iloc[i]['Date']}", key=f"del{i}"):
        data = data.drop(index=i)
        data.to_csv(data_file, index=False)
        st.rerun()

# --- Forecast Button ---
st.header("🔮 Forecast 10 Days Ahead")
if st.button("Run Forecast"):
    if len(data) < 5:
        st.warning("Need at least 5 data entries to generate forecast.")
    else:
        sm, cm, col = train_forecaster(data)
        forecast = make_forecast(sm, cm, col, data, event_data)
        forecast_df = pd.DataFrame(forecast, columns=["Date", "Forecasted Sales", "Forecasted Customers"])
        st.write(forecast_df)
        st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), "forecast.csv", "text/csv")
        st.line_chart(forecast_df.set_index("Date"))
