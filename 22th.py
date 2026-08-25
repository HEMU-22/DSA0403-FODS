import numpy as np
from scipy import stats

# Blood pressure reduction data for Drug group
drug = np.array([
    10, 12, 8, 15, 11,
    9, 13, 14, 10, 12,
    11, 13, 9, 15, 12,
    10, 14, 11, 13, 12,
    9, 10, 14, 13, 11
])

# Blood pressure reduction data for Placebo group
placebo = np.array([
    5, 7, 4, 8, 6,
    5, 7, 6, 4, 8,
    5, 6, 7, 5, 4,
    6, 8, 7, 5, 6,
    4, 5, 7, 6, 5
])

# Function to calculate 95% confidence interval
def confidence_interval(data):

    mean = np.mean(data)

    standard_error = stats.sem(data)

    confidence_interval = stats.t.interval(
        confidence=0.95,
        df=len(data) - 1,
        loc=mean,
        scale=standard_error
    )

    return mean, confidence_interval


# Drug group
drug_mean, drug_ci = confidence_interval(drug)

# Placebo group
placebo_mean, placebo_ci = confidence_interval(placebo)

# Display results
print("Drug Group")
print("Mean Reduction:", drug_mean)
print("95% Confidence Interval:", drug_ci)

print("\nPlacebo Group")
print("Mean Reduction:", placebo_mean)
print("95% Confidence Interval:", placebo_ci)