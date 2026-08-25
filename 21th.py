import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Data of 18 adults
age = [25, 30, 35, 40, 28, 32, 45, 50, 27,
       31, 36, 42, 29, 34, 38, 41, 26, 33]

fat = [18, 20, 22, 25, 19, 21, 27, 30, 17,
       23, 24, 26, 20, 22, 25, 28, 18, 21]

# Create DataFrame
data = pd.DataFrame({
    "Age": age,
    "%Fat": fat
})

# Calculate mean
print("Mean:")
print(data.mean())

# Calculate median
print("\nMedian:")
print(data.median())

# Calculate standard deviation
print("\nStandard Deviation:")
print(data.std())

# Boxplot
data.boxplot(column=["Age", "%Fat"])
plt.title("Boxplot of Age and Body Fat")
plt.show()

# Scatter plot
plt.scatter(data["Age"], data["%Fat"])
plt.title("Age vs Body Fat")
plt.xlabel("Age")
plt.ylabel("Body Fat (%)")
plt.grid(True)
plt.show()

# Q-Q plot for Age
stats.probplot(data["Age"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Age")
plt.show()

# Q-Q plot for Body Fat
stats.probplot(data["%Fat"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Body Fat")
plt.show()