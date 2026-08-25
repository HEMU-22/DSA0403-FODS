import pandas as pd

property_data = pd.DataFrame({
    "Property_ID": [101, 102, 103, 104, 105],
    "Location": ["Chennai", "Bangalore", "Chennai", "Hyderabad", "Bangalore"],
    "Bedrooms": [3, 5, 4, 6, 2],
    "Area": [1200, 2500, 1800, 3000, 1000],
    "Price": [5000000, 9000000, 7000000, 10000000, 4000000]
})

average_price = property_data.groupby("Location")["Price"].mean()

properties_more_than_4 = (property_data["Bedrooms"] > 4).sum()

largest_property = property_data.loc[property_data["Area"].idxmax()]

print("Average listing price by location:")
print(average_price)

print("\nNumber of properties with more than 4 bedrooms:")
print(properties_more_than_4)

print("\nProperty with the largest area:")
print(largest_property)