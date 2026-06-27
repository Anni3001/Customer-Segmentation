import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\anshu\Documents\Git Projects\Customer Segmentation\Database\Online Retail.csv")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nDataset Shape:", df.shape)

df = df.dropna(subset=["CustomerID", "Description"])

df = df.drop_duplicates()

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Total_Amount"] = df["Quantity"] * df["UnitPrice"]

cancelled = df["InvoiceNo"].str.startswith("C").sum()
print("\nCancelled Invoices:", cancelled)

df = df[~df["InvoiceNo"].str.startswith("C")]

negative_qty = (df["Quantity"] < 0).sum()
print("Negative Quantities:", negative_qty)

df = df[df["Quantity"] > 0]

zero_price = (df["UnitPrice"] <= 0).sum()
print("Zero or Negative Unit Prices:", zero_price)

df = df[df["UnitPrice"] > 0]

print("\nCleaned Dataset")
print(df.info())
print(df.shape)

customer_df = df.groupby("CustomerID").agg({
    "Total_Amount": "sum",
    "Quantity": "sum",
    "InvoiceNo": "nunique"
})

customer_df.columns = [
    "TotalSpent",
    "TotalQuantity",
    "NumTransactions"
]

customer_df = customer_df.reset_index()

print("\nCustomer Dataset")
print(customer_df.head())

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    customer_df[["TotalSpent", "TotalQuantity", "NumTransactions"]]
)

inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_features)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)

customer_df["Cluster"] = kmeans.fit_predict(scaled_features)

print("\nClustered Customers:")
print(customer_df.head())

print("\nCustomers in Each Cluster:")
print(customer_df["Cluster"].value_counts().sort_index())

cluster_summary = customer_df.groupby("Cluster").mean(numeric_only=True)

print("\nCluster Summary:")
print(cluster_summary)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    customer_df["TotalSpent"],
    customer_df["NumTransactions"],
    c=customer_df["Cluster"]
)

plt.xlabel("Total Spending")
plt.ylabel("Number of Transactions")
plt.title("Customer Segments")
plt.colorbar(scatter, label="Cluster")

plt.show()
