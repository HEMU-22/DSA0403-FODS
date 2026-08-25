import pandas as pd

# Create customer sales data
data = pd.DataFrame({
    "Age": [20, 25, 30, 25, 20, 35, 30, 25, 40, 30]
})

# Calculate frequency distribution
frequency = data["Age"].value_counts().sort_index()

# Display result
print("Frequency Distribution of Customer Ages:")
print(frequency)