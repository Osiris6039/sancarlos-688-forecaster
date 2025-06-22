
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from login_config import check_password

st.set_page_config(page_title="AI Sales & Customer Forecaster", layout="wide")

if not check_password():
    st.stop()

st.title("📊 AI Sales & Customer Forecaster")

@st.cache_data
def load_data():
    try:
        return pd.read_csv("historical_data.csv", parse_dates=["date"])
    except:
        return pd.DataFrame(columns=["date", "sales", "customers", "weather", "add_ons"])

def save_data(df):
    df = df.sort_values(by="date")
    df.to_csv("historical_data.csv", index=False)

data = load_data()

st.subheader("📝 Add New Data")
with st.form("data_form"):
    date = st.date_input("Date", value=datetime.date.today())
    sales = st.number_input("Sales", value=0)
    customers = st.number_input("Customers", value=0)
    weather = st.text_input("Weather")
    add_ons = st.number_input("Add-On Sales", value=0)
    submitted = st.form_submit_button("➕ Add Data")
    if submitted:
        new_row = pd.DataFrame([[date, sales, customers, weather, add_ons]], columns=data.columns)
        data = pd.concat([data, new_row], ignore_index=True)
        save_data(data)
        st.success("Data added successfully!")

st.subheader("📅 Historical Data")
delete_date = st.date_input("Select date to delete", value=None)
if st.button("🗑️ Delete Selected Date"):
    data = data[data["date"] != pd.to_datetime(delete_date)]
    save_data(data)
    st.success("Data deleted.")

st.dataframe(data)

# Future events
st.subheader("🎉 Add Future Event")
try:
    future_events = pd.read_csv("future_events.csv", parse_dates=["date"])
except:
    future_events = pd.DataFrame(columns=["date", "event", "last_year_sales", "last_year_customers"])

with st.form("future_event_form"):
    f_date = st.date_input("Future Event Date")
    f_event = st.text_input("Event Name")
    f_sales = st.number_input("Sales Last Year", value=0, key="ev_sales")
    f_customers = st.number_input("Customers Last Year", value=0, key="ev_customers")
    f_submit = st.form_submit_button("📌 Add Future Event")
    if f_submit:
        new_event = pd.DataFrame([[f_date, f_event, f_sales, f_customers]], columns=future_events.columns)
        future_events = pd.concat([future_events, new_event], ignore_index=True)
        future_events.to_csv("future_events.csv", index=False)
        st.success("Future event added!")

st.dataframe(future_events)

# Forecasting
st.subheader("📈 10-Day Forecast")
if len(data) > 20:
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    features = ["dayofweek", "month", "year", "customers", "add_ons"]
    X = df[features]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    forecast_dates = [df["date"].max() + datetime.timedelta(days=i) for i in range(1, 11)]

    future_df = pd.DataFrame({
        "date": forecast_dates,
        "dayofweek": [d.weekday() for d in forecast_dates],
        "month": [d.month for d in forecast_dates],
        "year": [d.year for d in forecast_dates],
        "customers": [int(df["customers"].mean())]*10,
        "add_ons": [0]*10
    })

    # Boost if future event exists
    for i, row in future_df.iterrows():
        match = future_events[future_events["date"] == row["date"]]
        if not match.empty:
            future_df.loc[i, "customers"] = match.iloc[0]["last_year_customers"]
            future_df.loc[i, "add_ons"] += int((match.iloc[0]["last_year_sales"] - df["sales"].mean()) * 0.5)

    prediction = model.predict(future_df[features])
    future_df["forecast_sales"] = prediction.astype(int)

    st.line_chart(future_df.set_index("date")[["forecast_sales"]])
    st.dataframe(future_df[["date", "forecast_sales"]])
    csv = future_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast", data=csv, file_name="forecast.csv", mime="text/csv")
else:
    st.warning("Please input at least 21 days of data.")
