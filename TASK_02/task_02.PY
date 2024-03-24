import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Step 2: Explore the dataset
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Display information about the dataset

# Step 3: Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Determine the optimal number of clusters (K)
# Here, we'll use the elbow method to find the optimal value of K
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Step 5: Build and train the K-means model with the chosen K
k = 5  # Chosen number of clusters based on the elbow method
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Step 6: Assign clusters to customers
data['Cluster'] = kmeans.labels_

# Step 7: Visualize the clusters
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='*', label='Centroids')
plt.title('Customer Segmentation based on Purchase Behavior')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Step 8: Interpret the results
# Analyze the characteristics of each cluster to understand differences between customer segments
