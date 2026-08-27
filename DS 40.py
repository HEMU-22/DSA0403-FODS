import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Synthetic dataset
data = {
    "CustomerID":[1,2,3,4,5,6,7,8],
    "TotalSpent":[200,400,600,800,1000,1200,1400,1600],
    "ItemsPurchased":[2,4,6,8,10,12,14,16]
}
df = pd.DataFrame(data)

X = df[["TotalSpent","ItemsPurchased"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

print("Customer Segments:\n", df)

# Visualization
plt.scatter(df["TotalSpent"], df["ItemsPurchased"], c=df["Cluster"], cmap="viridis")
plt.xlabel("Total Spent")
plt.ylabel("Items Purchased")
plt.title("Customer Segmentation Clusters")
plt.show()
