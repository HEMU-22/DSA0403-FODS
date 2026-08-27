import pandas as pd
from sklearn.cluster import KMeans

# Synthetic dataset
data = {
    "Purchases":[5,10,15,20,25,30],
    "Spending":[100,200,300,400,500,600]
}
df = pd.DataFrame(data)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

print(df)

# User input
new_customer = pd.DataFrame([[18,350]], columns=["Purchases","Spending"])
cluster = kmeans.predict(new_customer)[0]
print("New customer belongs to cluster:", cluster)
