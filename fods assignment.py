# ============================================================
# HOUSE PRICE PREDICTION AND PROPERTY MARKET SEGMENTATION
# SECTION 6 - DESCRIPTIVE STATISTICAL ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("\n==============================================")
print("       CALIFORNIA HOUSING DATASET")
print("==============================================")

housing = fetch_california_housing(as_frame=True)

df = housing.frame

# Rename target column
df.rename(columns={"MedHouseVal": "MedHouseVal"}, inplace=True)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ------------------------------------------------------------
# 2. BASIC INFORMATION
# ------------------------------------------------------------

print("\n==============================================")
print("             DATA INFORMATION")
print("==============================================")

print(df.info())


# ------------------------------------------------------------
# 3. MISSING VALUE CHECK
# ------------------------------------------------------------

print("\n==============================================")
print("             MISSING VALUES")
print("==============================================")

print(df.isnull().sum())


# ------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n==============================================")
print("       DESCRIPTIVE STATISTICS")
print("==============================================")

print("\nMean:")
print(df.mean())

print("\nVariance:")
print(df.var())

print("\nStandard Deviation:")
print(df.std())

print("\nMinimum:")
print(df.min())

print("\nMaximum:")
print(df.max())


# ------------------------------------------------------------
# 5. MEAN OF HOUSE PRICE
# ------------------------------------------------------------

mean_price = df["MedHouseVal"].mean()

print("\n==============================================")
print("          HOUSE PRICE MEAN")
print("==============================================")

print("Mean House Value:", round(mean_price, 4))


# ------------------------------------------------------------
# 6. VARIANCE OF HOUSE PRICE
# ------------------------------------------------------------

variance_price = df["MedHouseVal"].var()

print("\n==============================================")
print("        HOUSE PRICE VARIANCE")
print("==============================================")

print("Variance of House Value:", round(variance_price, 4))


# ------------------------------------------------------------
# 7. COVARIANCE MATRIX
# ------------------------------------------------------------

print("\n==============================================")
print("             COVARIANCE MATRIX")
print("==============================================")

covariance_matrix = df.cov()

print(covariance_matrix.round(4))


# ------------------------------------------------------------
# 8. CORRELATION MATRIX
# ------------------------------------------------------------

print("\n==============================================")
print("             CORRELATION MATRIX")
print("==============================================")

correlation_matrix = df.corr()

print(correlation_matrix.round(4))


# ------------------------------------------------------------
# 9. CORRELATION WITH HOUSE PRICE
# ------------------------------------------------------------

print("\n==============================================")
print("   CORRELATION WITH HOUSE PRICE")
print("==============================================")

price_correlation = (
    correlation_matrix["MedHouseVal"]
    .sort_values(ascending=False)
)

print(price_correlation.round(4))


# ------------------------------------------------------------
# 10. STRONGEST POSITIVE RELATIONSHIP
# ------------------------------------------------------------

print("\n==============================================")
print("       STRONGEST PRICE RELATIONSHIP")
print("==============================================")

price_correlation_without_target = (
    price_correlation.drop("MedHouseVal")
)

strongest_attribute = price_correlation_without_target.idxmax()
strongest_value = price_correlation_without_target.max()

print("Attribute:", strongest_attribute)
print("Correlation:", round(strongest_value, 4))


# ------------------------------------------------------------
# 11. PRICE DISTRIBUTION GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["MedHouseVal"], bins=40)

plt.title("Distribution of House Prices")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12. IMPORTANT ATTRIBUTE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["MedInc"], bins=40)

plt.title("Distribution of Median Income")
plt.xlabel("Median Income")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 13. HOUSE PRICE VS MEDIAN INCOME
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["MedInc"],
    df["MedHouseVal"],
    alpha=0.3
)

plt.title("Median Income vs House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 14. CORRELATION BAR GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

corr_values = price_correlation.drop("MedHouseVal")

plt.bar(
    corr_values.index,
    corr_values.values
)

plt.title("Correlation of Attributes with House Price")
plt.xlabel("Attributes")
plt.ylabel("Correlation")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==============================================")
print("           STATISTICAL SUMMARY")
print("==============================================")

print("Total Records       :", len(df))
print("Total Attributes    :", len(df.columns))
print("Mean House Value    :", round(mean_price, 4))
print("House Value Variance:", round(variance_price, 4))
print("Strongest Attribute :", strongest_attribute)
print("Correlation Value   :", round(strongest_value, 4))

print("\n==============================================")
print("     DESCRIPTIVE ANALYSIS COMPLETED")
print("==============================================")
