import pandas as pd

# Create post likes data
data = pd.DataFrame({
    "Likes": [10, 20, 10, 30, 20, 10, 40, 30, 20, 10]
})

# Calculate frequency distribution
frequency = data["Likes"].value_counts().sort_index()

# Display result
print("Frequency Distribution of Likes:")
print(frequency)