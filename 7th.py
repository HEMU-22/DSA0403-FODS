import pandas as pd

order_data = pd.DataFrame({
    "Customer_ID": [101, 102, 101, 103, 102, 101],
    "Order_Date": [
        "2026-08-01", "2026-08-03", "2026-08-05",
        "2026-08-07", "2026-08-10", "2026-08-12"
    ],
    "Product": [
        "Laptop", "Phone", "Laptop",
        "Tablet", "Phone", "Laptop"
    ],
    "Quantity": [2, 1, 3, 2, 4, 1]
})

order_data["Order_Date"] = pd.to_datetime(order_data["Order_Date"])

orders_by_customer = order_data.groupby("Customer_ID").size()

average_quantity = order_data.groupby("Product")["Quantity"].mean()

earliest_date = order_data["Order_Date"].min()
latest_date = order_data["Order_Date"].max()

print("Total orders by each customer:")
print(orders_by_customer)

print("\nAverage order quantity for each product:")
print(average_quantity)

print("\nEarliest order date:", earliest_date)
print("Latest order date:", latest_date)