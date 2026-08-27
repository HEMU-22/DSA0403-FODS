import pandas as pd
from sklearn.cluster import KMeans

# Synthetic dataset
data = {
    "CustomerID":[1,2,3,4,5,6],
    "TotalSpent":[200,400,600,800,1000,1200],
    "VisitFrequency":[2,4,6,8,10,12]
}
df = pd.DataFrame(data)

X = df[["TotalSpent","VisitFrequency"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

print("Customer Segments:\n", df)
