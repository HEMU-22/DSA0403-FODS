import pandas as pd
from sklearn.cluster import KMeans

# Synthetic dataset
data = {
    "PurchaseHistory":[10,20,30,40,50,60],
    "BrowsingTime":[5,10,15,20,25,30],
    "Age":[25,30,35,40,45,50]
}
df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

print("Customer Segments:\n", df)
