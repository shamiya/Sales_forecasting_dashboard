import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load inventory and sales data
inventory_df = pd.read_excel("inventory_details_infos.xlsx")
sales_df = pd.read_excel("stock_items.xlsx")

# Merge data on item_id (inventory) and id (sales)
merged_df = pd.merge(sales_df, inventory_df, left_on='id', right_on='item_id', how='left')

# Convert date columns to datetime
merged_df['created_at_x'] = pd.to_datetime(merged_df['created_at_x'])

# Streamlit App Title
st.title("Sales and Inventory Dashboard")

# Monthly Sales Trend
st.subheader("Monthly Sales Trend")
sales_trend = merged_df.groupby(merged_df['created_at_x'].dt.to_period("M"))['amount'].sum()
fig, ax = plt.subplots()
sales_trend.plot(kind='line', marker='o', ax=ax)
ax.set_title("Monthly Sales Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Sales Amount")
st.pyplot(fig)

# Inventory Analysis
st.subheader("Low Stock Items")
low_stock = merged_df[['item_name', 'current_stock_y']].sort_values(by="current_stock_y").head(10)
st.write(low_stock)

# ARIMA Forecasting
st.subheader("Sales Forecast (Next 6 Months)")
if len(sales_trend) > 6:
    model = ARIMA(sales_trend, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)
    forecast_df = pd.DataFrame({"Month": range(1, 7), "Forecasted Sales": forecast})
    st.write(forecast_df)
else:
    st.write("Not enough data for forecasting.")

st.write("Developed by Your Name")
