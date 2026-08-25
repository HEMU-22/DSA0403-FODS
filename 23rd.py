import numpy as np
from scipy.stats import ttest_ind

# Conversion rate data for Design A
A = np.array([
    0.12, 0.15, 0.10, 0.14, 0.13,
    0.11, 0.16, 0.12, 0.14, 0.13
])

# Conversion rate data for Design B
B = np.array([
    0.18, 0.20, 0.17, 0.19, 0.21,
    0.18, 0.22, 0.20, 0.19, 0.21
])

# Calculate mean conversion rates
mean_A = np.mean(A)
mean_B = np.mean(B)

# Perform independent t-test
t_stat, p_value = ttest_ind(A, B)

# Display results
print("Mean Conversion Rate - Design A:", mean_A)
print("Mean Conversion Rate - Design B:", mean_B)

print("t-statistic:", t_stat)
print("p-value:", p_value)

# Significance level
alpha = 0.05

# Decision
if p_value < alpha:
    print("There is a statistically significant difference.")
else:
    print("There is no statistically significant difference.")