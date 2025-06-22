
import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# App config
st.set_page_config(page_title="Smart Sales & Customer Forecaster", layout="wide")

# Authentication
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# Load or initialize historical dataset
DATA_FILE = "historical_data.csv"
FUTURE_EVENTS_FILE = "future_events.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
        df.sort_values("Date", inplace=True)
        return df
    return pd.DataFrame(columns=["Date", "Sales", "Customers", "Weather", "Add_on_Sales"])

def load_events():
    if os.path.exists(FUTURE_EVENTS_FILE):
        df = pd.read_csv(FUTURE_EVENTS_FILE, parse_dates=["Event_Date"])
        df.sort_values("Event_Date", inplace=True)
        return df
    return pd.DataFrame(columns=["Event_Date", "Event_Name", "Past_Sales", "Past_Customers"])

def save_data(df):
    df.sort_values("Date", inplace=True)
    df.to_csv(DATA_FILE, index=False)

def save_events(df):
    df.sort_values("Event_Date", inplace=True)
    df.to_csv(FUTURE_EVENTS_FILE, index=False)

df = load_data()
events = load_events()

# Sidebar navigation
menu = st.sidebar.radio("📌 Menu", ["📈 Forecast", "➕ Input Data", "📅 Future Events"])

# Forecasting function
def train_and_forecast(df, events):
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Add_on_Sales"] = df["Add_on_Sales"].fillna(0)
    df["Weather"] = df["Weather"].fillna("Unknown")

    le = LabelEncoder()
    df["Weather_Code"] = le.fit_transform(df["Weather"])

    features = ["DayOfWeek", "Month", "Weather_Code", "Add_on_Sales"]
    X_sales = df[features]
    y_sales = df["Sales"]
    y_customers = df["Customers"]

    model_sales = XGBRegressor(n_estimators=100)
    model_customers = XGBRegressor(n_estimators=100)
    model_sales.fit(X_sales, y_sales)
    model_customers.fit(X_sales, y_customers)

    future_dates = pd.date_range(datetime.date.today(), periods=10)
    future_df = pd.DataFrame({
        "Date": future_dates,
        "DayOfWeek": future_dates.dayofweek,
        "Month": future_dates.month,
        "Weather_Code": 0,
        "Add_on_Sales": 0
    })

    # Match events
    for i, row in future_df.iterrows():
        event_row = events[events["Event_Date"] == row["Date"]]
        if not event_row.empty:
            future_df.loc[i, "Add_on_Sales"] = event_row["Past_Sales"].values[0] * 0.25

    sales_pred = model_sales.predict(future_df[features])
    customers_pred = model_customers.predict(future_df[features])
    result = future_df[["Date"]].copy()
    result["Forecasted_Sales"] = sales_pred.astype(int)
    result["Forecasted_Customers"] = customers_pred.astype(int)
    return result

# Forecast tab
if menu == "📈 Forecast":
    st.subheader("📊 10-Day Forecast for Sales & Customers")
    forecast = train_and_forecast(df, events)
    st.dataframe(forecast)

    st.line_chart(forecast.set_index("Date")[["Forecasted_Sales", "Forecasted_Customers"]])
    st.download_button("⬇️ Download Forecast CSV", forecast.to_csv(index=False), "forecast.csv")

# Data input tab
elif menu == "➕ Input Data":
    st.subheader("📝 Add New Daily Data")
    with st.form("input_form"):
        date = st.date_input("Date", value=datetime.date.today())
        sales = st.number_input("Sales", min_value=0)
        customers = st.number_input("Customers", min_value=0)
        weather = st.text_input("Weather")
        add_on = st.number_input("Add-on Sales", min_value=0)
        submitted = st.form_submit_button("Save")
        if submitted:
            new_row = {"Date": date, "Sales": sales, "Customers": customers, "Weather": weather, "Add_on_Sales": add_on}
            df = df.append(new_row, ignore_index=True)
            save_data(df)
            st.success("Data added successfully!")

    if not df.empty:
        st.write("🗑️ Delete an entry:")
        selected_date = st.selectbox("Select Date to Delete", df["Date"].astype(str))
        if st.button("Delete Entry"):
            df = df[df["Date"].astype(str) != selected_date]
            save_data(df)
            st.warning(f"Entry for {selected_date} deleted!")

        st.dataframe(df)

# Future Events tab
elif menu == "📅 Future Events":
    st.subheader("🎯 Add Reference for Future Events")
    with st.form("event_form"):
        event_date = st.date_input("Event Date")
        event_name = st.text_input("Event Name (e.g. Charter Day)")
        past_sales = st.number_input("Last Year's Sales", min_value=0)
        past_customers = st.number_input("Last Year's Customers", min_value=0)
        submitted = st.form_submit_button("Save Event")
        if submitted:
            new_event = {"Event_Date": event_date, "Event_Name": event_name, "Past_Sales": past_sales, "Past_Customers": past_customers}
            events = events.append(new_event, ignore_index=True)
            save_events(events)
            st.success("Event saved!")

    if not events.empty:
        st.write("📅 Future Events Reference")
        st.dataframe(events)
        selected_event = st.selectbox("Select Event Date to Delete", events["Event_Date"].astype(str))
        if st.button("Delete Event"):
            events = events[events["Event_Date"].astype(str) != selected_event]
            save_events(events)
            st.warning(f"Event on {selected_event} deleted.")
