# ============================================================
# HOUSE PRICE PREDICTION AND PROPERTY MARKET SEGMENTATION
# SECTIONS 7, 8 AND 9
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy import stats


# ============================================================
# 1. LOAD DATASET - OFFLINE CACHE
# ============================================================

print("\n==============================================")
print("       HOUSE PRICE PREDICTION SYSTEM")
print("==============================================")

print("\nLoading California Housing Dataset...")

housing = fetch_california_housing(
    as_frame=True,
    download_if_missing=False
)

df = housing.frame

print("Dataset loaded successfully.")

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 2. BASIC DATA CHECK
# ============================================================

print("\n==============================================")
print("             DATA CHECK")
print("==============================================")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


# ============================================================
# 3. STATISTICAL INFERENCE
#    95% CONFIDENCE INTERVAL
# ============================================================

print("\n==============================================")
print("       7. STATISTICAL INFERENCE")
print("==============================================")

price = df["MedHouseVal"]

sample_mean = price.mean()
sample_std = price.std()
sample_size = len(price)

confidence_level = 0.95
alpha = 1 - confidence_level

standard_error = sample_std / np.sqrt(sample_size)

t_value = stats.t.ppf(
    1 - alpha / 2,
    sample_size - 1
)

margin_of_error = t_value * standard_error

lower_limit = sample_mean - margin_of_error
upper_limit = sample_mean + margin_of_error

print("\nSample Mean:")
print(round(sample_mean, 4))

print("\nSample Standard Deviation:")
print(round(sample_std, 4))

print("\nSample Size:")
print(sample_size)

print("\n95% CONFIDENCE INTERVAL")
print("----------------------------------------------")

print("Lower Limit:", round(lower_limit, 4))
print("Upper Limit:", round(upper_limit, 4))

print("\nInterpretation:")
print(
    "The 95% confidence interval represents the range "
    "within which the population mean house value is "
    "expected to lie with 95% confidence."
)


# ============================================================
# 4. FEATURE AND TARGET SELECTION
# ============================================================

print("\n==============================================")
print("      8. FEATURE AND TARGET SELECTION")
print("==============================================")

X = df.drop("MedHouseVal", axis=1)

y = df["MedHouseVal"]

print("\nIndependent Variables (X):")

for column in X.columns:
    print("-", column)

print("\nDependent Variable (y):")
print("- MedHouseVal")


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n==============================================")
print("             TRAIN TEST SPLIT")
print("==============================================")

print("Training Records:", len(X_train))
print("Testing Records :", len(X_test))


# ============================================================
# 6. CREATE MODELS
# ============================================================

print("\n==============================================")
print("          CREATING THREE MODELS")
print("==============================================")

# Linear Regression
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# kNN Regression
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=5))
])

# Decision Tree Regression
decision_tree_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=10
)

print("\n1. Linear Regression")
print("2. kNN Regression")
print("3. Decision Tree Regression")


# ============================================================
# 7. TRAIN MODELS
# ============================================================

print("\n==============================================")
print("             MODEL TRAINING")
print("==============================================")

linear_model.fit(X_train, y_train)

print("Linear Regression trained.")

knn_model.fit(X_train, y_train)

print("kNN Regression trained.")

decision_tree_model.fit(X_train, y_train)

print("Decision Tree Regression trained.")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

print("\n==============================================")
print("             PRICE PREDICTION")
print("==============================================")

linear_pred = linear_model.predict(X_test)

knn_pred = knn_model.predict(X_test)

tree_pred = decision_tree_model.predict(X_test)

print("Predictions generated successfully.")


# ============================================================
# 9. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predicted
    )

    return [
        name,
        mae,
        mse,
        rmse,
        r2
    ]


# ============================================================
# 10. EVALUATE THREE MODELS
# ============================================================

print("\n==============================================")
print("      9. MODEL PERFORMANCE EVALUATION")
print("==============================================")

results = []

results.append(
    evaluate_model(
        "Linear Regression",
        y_test,
        linear_pred
    )
)

results.append(
    evaluate_model(
        "kNN Regression",
        y_test,
        knn_pred
    )
)

results.append(
    evaluate_model(
        "Decision Tree Regression",
        y_test,
        tree_pred
    )
)


# ============================================================
# 11. CREATE PERFORMANCE TABLE
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ]
)

print("\n==============================================")
print("       MODEL PERFORMANCE COMPARISON")
print("==============================================")

print(
    results_df.round(4).to_string(index=False)
)


# ============================================================
# 12. SELECT BEST MODEL
# ============================================================

best_index = results_df["RMSE"].idxmin()

best_model = results_df.loc[best_index]

best_model_name = best_model["Model"]

print("\n==============================================")
print("             BEST MODEL")
print("==============================================")

print("Best Model :", best_model_name)

print(
    "MAE        :",
    round(best_model["MAE"], 4)
)

print(
    "MSE        :",
    round(best_model["MSE"], 4)
)

print(
    "RMSE       :",
    round(best_model["RMSE"], 4)
)

print(
    "R2 Score   :",
    round(best_model["R2 Score"], 4)
)

print("\nReason:")
print(
    "The model with the lowest RMSE is selected "
    "as the best-performing model."
)


# ============================================================
# 13. SELECT BEST PREDICTIONS
# ============================================================

if best_model_name == "Linear Regression":

    best_predictions = linear_pred

elif best_model_name == "kNN Regression":

    best_predictions = knn_pred

else:

    best_predictions = tree_pred


# ============================================================
# 14. ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\nGenerating Actual vs Predicted graph...")

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.4
)

plt.xlabel("Actual House Value")

plt.ylabel("Predicted House Value")

plt.title(
    "Actual vs Predicted House Values"
)

plt.tight_layout()

plt.show()


# ============================================================
# 15. RMSE MODEL COMPARISON GRAPH
# ============================================================

print("\nGenerating RMSE comparison graph...")

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.xlabel("Regression Models")

plt.ylabel("RMSE")

plt.title(
    "Model Comparison Using RMSE"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# ============================================================
# 16. R2 MODEL COMPARISON GRAPH
# ============================================================

print("\nGenerating R2 comparison graph...")

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.xlabel("Regression Models")

plt.ylabel("R2 Score")

plt.title(
    "Model Comparison Using R2 Score"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# ============================================================
# 17. ACTUAL AND PREDICTED TABLE
# ============================================================

comparison_df = pd.DataFrame({
    "Actual Price": y_test.values[:20],
    "Predicted Price": best_predictions[:20]
})

print("\n==============================================")
print("      ACTUAL VS PREDICTED VALUES")
print("==============================================")

print(
    comparison_df.round(4).to_string(index=False)
)


# ============================================================
# 18. SAVE MODEL RESULTS
# ============================================================

results_df.to_csv(
    "model_performance_results.csv",
    index=False
)

comparison_df.to_csv(
    "actual_vs_predicted_results.csv",
    index=False
)

print("\nResult files saved successfully.")


# ============================================================
# 19. FINAL STATUS
# ============================================================

print("\n==============================================")
print("       SECTIONS 7, 8 AND 9 COMPLETED")
print("==============================================")

print("Statistical Inference       : COMPLETED")
print("95% Confidence Interval     : COMPLETED")
print("Feature Selection           : COMPLETED")
print("Train-Test Split            : COMPLETED")
print("Linear Regression           : COMPLETED")
print("kNN Regression              : COMPLETED")
print("Decision Tree Regression    : COMPLETED")
print("MAE                         : CALCULATED")
print("MSE                         : CALCULATED")
print("RMSE                        : CALCULATED")
print("R2 Score                    : CALCULATED")
print("Best Model Selection        : COMPLETED")
print("Actual vs Predicted         : GENERATED")
print("RMSE Comparison             : GENERATED")
print("R2 Comparison               : GENERATED")

print("\n==============================================")
print("             PROGRAM COMPLETED")
print("==============================================")
