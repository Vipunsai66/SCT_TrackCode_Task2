import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load CSV data
df = pd.read_csv('customers.csv')

print("Sample of the data:")
print(df.head())

# 2. Select relevant features
features = ['Age', 'AnnualIncome(k$)', 'SpendingScore(1-100)']
data = df[features]

# 3. Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 4. Find optimal k using the elbow method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# 5. Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method to Determine Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# 6. Apply KMeans with selected k (e.g., 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# 7. Visualize clusters
sns.pairplot(df, hue='Cluster', vars=features, palette='Set2')
plt.suptitle("Customer Segments", y=1.02)
plt.show()

# 8. Save results to a new CSV
df.to_csv('clustered_customers.csv', index=False)

print("\nClustered Data Sample:")
print(df[['CustomerID', 'Cluster'] + features])
