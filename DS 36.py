import pandas as pd
from sklearn.linear_model import LogisticRegression

# Synthetic dataset
data = {
    "UsageMinutes":[300,150,400,100,250,500],
    "ContractDuration":[12,6,24,3,12,18],
    "Churn":[0,1,0,1,0,1]  # 0=Not churned, 1=Churned
}
df = pd.DataFrame(data)

X = df[["UsageMinutes","ContractDuration"]]
y = df["Churn"]

model = LogisticRegression()
model.fit(X, y)

# User input
new_customer = pd.DataFrame([[350,12]], columns=["UsageMinutes","ContractDuration"])
pred = model.predict(new_customer)[0]

print("Predicted Churn:", "Yes" if pred==1 else "No")
