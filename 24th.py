import pandas as pd
import numpy as np
from scipy import stats

# Read CSV file
data = pd.read_csv("rare_elements.csv")

# User inputs
sample_size = int(input("Enter sample size: "))
confidence = float(input("Enter confidence level (%): "))
precision = float(input("Enter desired precision: "))

# Select sample
sample = data.iloc[:sample_size, 0].dropna()

# Calculate sample mean
mean = np.mean(sample)

# Convert confidence level to decimal
confidence_level = confidence / 100

# Calculate standard error
standard_error = stats.sem(sample)

# Calculate t critical value
t_value = stats.t.ppf(
    (1 + confidence_level) / 2,
    len(sample) - 1
)

# Calculate margin of error
margin_error = t_value * standard_error

# Calculate confidence interval
lower = mean - margin_error
upper = mean + margin_error

# Display results
print("\nPoint Estimate:", mean)

print("Standard Error:", standard_error)

print("Margin of Error:", margin_error)

print("Confidence Interval:",
      (lower, upper))

# Check desired precision
if margin_error <= precision:
    print("Desired precision is achieved.")
else:
    print("Desired precision is not achieved.")
    print("Increase the sample size.")