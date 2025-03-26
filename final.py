import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
# Load inventory data from Excel
inventory_df = pd.read_excel("inventory_details_infos.xlsx")  # Correct function

# Load sales data from Excel
sales_df = pd.read_excel("stock_items.xlsx")  # Update filename if needed

# Merge sales and inventory data on item_id from inventory_df and id from sales_df
merged_df = pd.merge(sales_df, inventory_df, left_on='id', right_on='item_id', how='left')

### Exploratory Data Analysis (EDA)

merged_df.groupby(merged_df['created_at_x'].dt.to_period("M"))['amount'].sum().plot(kind='line', marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.show()

###. Analyze Inventory Stock Levels
(merged_df[['item_name', 'current_stock_y']].sort_values(by="current_stock_y").head(100))

# Aggregate sales data by date
sales_trend = merged_df.groupby(merged_df['created_at_x'].dt.to_period("M"))['amount'].sum()

# Fit ARIMA model
model = ARIMA(sales_trend, order=(5,1,0))
model_fit = model.fit()

# Forecast next 6 months
forecast = model_fit.forecast(steps=6)
print(forecast)
