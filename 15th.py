import pandas as pd

# Temperature data for different cities
data = pd.DataFrame({
    "Chennai": [30, 32, 31, 35, 34, 33],
    "Delhi": [20, 25, 30, 35, 40, 45],
    "Mumbai": [28, 29, 30, 29, 28, 30]
})

# 1. Calculate mean temperature for each city
mean_temperature = data.mean()

# 2. Calculate standard deviation for each city
standard_deviation = data.std()

# 3. Calculate temperature range
temperature_range = data.max() - data.min()

print("Mean Temperature:")
print(mean_temperature)

print("\nStandard Deviation:")
print(standard_deviation)

print("\nTemperature Range:")
print(temperature_range)

# City with highest temperature range
highest_range_city = temperature_range.idxmax()

# City with lowest standard deviation
most_consistent_city = standard_deviation.idxmin()

print("\nCity with Highest Temperature Range:",
      highest_range_city)

print("Most Consistent City:",
      most_consistent_city)