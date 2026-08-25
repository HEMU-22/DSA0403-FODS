import pandas as pd
from scipy import stats
import numpy as np

# Read customer reviews data
data = pd.read_csv("customer_reviews.csv")

# Select rating column
ratings = data["Rating"].dropna()

# Calculate sample mean
mean = ratings.mean()

# Calculate standard error
standard_error = stats.sem(ratings)

# Calculate 95% confidence interval
confidence_interval = stats.t.interval(
    0.95,
    df=len(ratings) - 1,
    loc=mean,
    scale=standard_error
)

# Display results
print("Mean Customer Rating:", mean)

print("95% Confidence Interval:")
print("Lower Limit:", confidence_interval[0])
print("Upper Limit:", confidence_interval[1])