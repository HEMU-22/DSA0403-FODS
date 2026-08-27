import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Synthetic dataset
data = {
    "EngineSize":[1.2,1.5,2.0,2.5,3.0],
    "Horsepower":[100,120,150,180,200],
    "FuelEfficiency":[20,18,15,12,10],
    "Price":[10000,15000,20000,25000,30000]
}
df = pd.DataFrame(data)

X = df[["EngineSize","Horsepower","FuelEfficiency"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("MSE:", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))

# Predict new car
new_car = pd.DataFrame([[2.2,160,14]], columns=["EngineSize","Horsepower","FuelEfficiency"])
print("Predicted Price:", model.predict(new_car)[0])
