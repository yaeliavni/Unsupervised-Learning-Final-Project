"""this code both checks for outliers and after recieving the noise cluster it looks into that
one as well to further explore the relation if exists to outliers at all
from our observation of all the project there aren't any ourliers that are real data entry mistakes
in order for it recognizes real data points that just are at a higher clinial risk (and that is shown in their data)
or that are just not clustered well because they belong in two different clusters so they are shown on the edge and such else"""
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, mannwhitneyu

import umap
import hdbscan
import warnings

warnings.filterwarnings('ignore')

# --- File Path ---
DATA_PATH = r"C:\\Users\\avni1\\Documents\\שנה ג\\unsupervised learning\\Final Data For Project\\parkinsons_disease_data.csv"

# --- Load Data ---
df = pd.read_csv(DATA_PATH)

# --- Step 0: HDBSCAN Clustering Automatically ---
print("Running HDBSCAN Clustering to create 'cluster' labels...")
features = df.select_dtypes(include=[np.number]).columns

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Perform HDBSCAN Clustering
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
cluster_labels = hdb.fit_predict(X_scaled)

# Add Cluster Labels to the DataFrame
df['cluster'] = cluster_labels

# --- Separate Noise Points ---
df_noise = df[df['cluster'] == -1]
df_clustered = df[df['cluster'] != -1]

print(f"Number of Noise Points: {len(df_noise)}")
print(f"Number of Clustered Points: {len(df_clustered)}")

# --- Focus Only on Numerical Features (excluding 'cluster')
features = df.select_dtypes(include=[np.number]).drop(columns=['cluster']).columns

# --- Step 1: Compare Feature Distributions (Boxplots) ---
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=(df['cluster'] == -1), y=feature)
    plt.title(f"Distribution of {feature} - Noise (True/False)")
    plt.xlabel('Is Noise Point')
    plt.ylabel(feature)
    plt.grid(True)
    plt.show()

# --- Step 2: Statistical Testing Feature-by-Feature ---
print("\n=== Statistical Testing (Noise vs Clustered) ===")
results = []
for feature in features:
    noise_vals = df_noise[feature].dropna()
    cluster_vals = df_clustered[feature].dropna()
    if len(noise_vals) > 0 and len(cluster_vals) > 0:
        stat, p_val = mannwhitneyu(noise_vals, cluster_vals, alternative='two-sided')
        results.append((feature, p_val))

# Sort by p-value
results_sorted = sorted(results, key=lambda x: x[1])

print("\nFeature | Mann-Whitney p-value (Noise vs Clustered)")
for feature, p_val in results_sorted:
    print(f"{feature}: p = {p_val:.4e}")

# --- Step 3: UMAP Visualization ---
print("\nGenerating UMAP Plot...")
X = df[features].fillna(0)
X_scaled = scaler.fit_transform(X)  # re-scaling after dropping 'cluster'

reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=df['cluster'], palette='tab20', legend=None)
plt.title('UMAP Projection with Noise Highlighted')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.grid(True)
plt.show()

# --- Step 4: Outlier Detection Only on Noise Points ---
print("\nRunning Outlier Detection inside Noise Points...")

X_noise = df_noise[features].fillna(0)
X_noise_scaled = scaler.transform(X_noise)

# 1. Isolation Forest
iso = IsolationForest(contamination=0.1, random_state=42)
iso_outliers = iso.fit_predict(X_noise_scaled)

# 2. Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20)
lof_outliers = lof.fit_predict(X_noise_scaled)

# 3. One-Class SVM
svm = OneClassSVM(kernel='rbf', nu=0.1)
svm_outliers = svm.fit_predict(X_noise_scaled)

# --- Print Outlier Detection Summary ---
print(f"Isolation Forest flagged {np.sum(iso_outliers == -1)} noise points as outliers.")
print(f"Local Outlier Factor flagged {np.sum(lof_outliers == -1)} noise points as outliers.")
print(f"One-Class SVM flagged {np.sum(svm_outliers == -1)} noise points as outliers.")

# --- Visualize Outliers inside Noise (optional) ---
pca = PCA(n_components=2, random_state=42)
X_noise_pca = pca.fit_transform(X_noise_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_noise_pca[:, 0], X_noise_pca[:, 1], c=(iso_outliers == -1), cmap='coolwarm', edgecolor='k')
plt.title('PCA of Noise Points - Isolation Forest Outliers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

# --- Step 5: EXTRA Unsupervised Learning based Outlier Scoring ---
print("\nRunning Unsupervised Model: Learning Normal Structure from Clustered Points...")

# Train IsolationForest on clustered points only
X_clustered = df_clustered[features].fillna(0)
X_clustered_scaled = scaler.transform(X_clustered)

unsupervised_model = IsolationForest(contamination=0.1, random_state=42)
unsupervised_model.fit(X_clustered_scaled)

# Score noise points
noise_scores = unsupervised_model.decision_function(X_noise_scaled)

# Decision function: higher = more normal, lower = more anomalous
print("\nNoise Points Outlier Scores (Higher = more normal, Lower = more anomalous):")
print(noise_scores)

# Plot Noise Scores
plt.figure(figsize=(8, 4))
sns.histplot(noise_scores, kde=True, bins=30)
plt.title('Outlier Scores of Noise Points (Learned from Clustered Patients)')
plt.xlabel('Outlier Score')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Optional: print top 10 most "abnormal" noise points
most_abnormal_idx = np.argsort(noise_scores)[:10]
print("\nTop 10 Most Abnormal Noise Points (Indices):")
print(df_noise.iloc[most_abnormal_idx])
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

import umap
import hdbscan
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = r"C:\\Users\\avni1\\Documents\\שנה ג\\unsupervised learning\\Final Data For Project\\parkinsons_disease_data.csv"
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["PatientID"], errors="ignore")

print("Running global outlier detection methods on entire dataset...")

features = df.select_dtypes(include=[np.number]).columns
X = df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(X_scaled)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(X_scaled)

svm = OneClassSVM(kernel='rbf', nu=0.05)
svm_labels = svm.fit_predict(X_scaled)

df['Outlier_IsolationForest'] = iso_labels
df['Outlier_LOF'] = lof_labels
df['Outlier_OCSVM'] = svm_labels
print("\nVisualizing results of global outlier detection methods...")

outlier_methods = {
    'Outlier_IsolationForest': iso_labels,
    'Outlier_LOF': lof_labels,
    'Outlier_OCSVM': svm_labels
}

print("\nGenerating UMAP Plots for Global Outlier Detection...")

reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

outlier_methods = {
    'Outlier_IsolationForest': df['Outlier_IsolationForest'],
    'Outlier_LOF': df['Outlier_LOF'],
    'Outlier_OCSVM': df['Outlier_OCSVM']
}

for method_name, labels in outlier_methods.items():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=(labels == -1), palette={False: 'gray', True: 'red'}, legend=False)
    plt.title(f'UMAP Projection - Outliers Highlighted ({method_name})')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.grid(True)
    plt.show()

print(f"Isolation Forest flagged {np.sum(iso_labels == -1)} global outliers.")
print(f"Local Outlier Factor flagged {np.sum(lof_labels == -1)} global outliers.")
print(f"One-Class SVM flagged {np.sum(svm_labels == -1)} global outliers.")

print("\nRunning HDBSCAN Clustering to create 'cluster' labels...")
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
cluster_labels = hdb.fit_predict(X_scaled)
df['cluster'] = cluster_labels
df_noise = df[df['cluster'] == -1]
df_clustered = df[df['cluster'] != -1]

print(f"Number of Noise Points: {len(df_noise)}")
print(f"Number of Clustered Points: {len(df_clustered)}")
features = df.select_dtypes(include=[np.number]).drop(columns=[
    'cluster', 'Outlier_IsolationForest', 'Outlier_LOF', 'Outlier_OCSVM'
]).columns

"""for feature in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=(df['cluster'] == -1), y=feature)
    plt.title(f"Distribution of {feature} - Noise (True/False)")
    plt.xlabel('Is Noise Point')
    plt.ylabel(feature)
    plt.grid(True)
    plt.show()"""
print("\n=== Statistical Testing (Noise vs Clustered) ===")
results = []
for feature in features:
    noise_vals = df_noise[feature].dropna()
    cluster_vals = df_clustered[feature].dropna()
    if len(noise_vals) > 0 and len(cluster_vals) > 0:
        stat, p_val = mannwhitneyu(noise_vals, cluster_vals, alternative='two-sided')
        results.append((feature, p_val))
results_sorted = sorted(results, key=lambda x: x[1])
print("\nFeature | Mann-Whitney p-value (Noise vs Clustered)")
for feature, p_val in results_sorted:
    print(f"{feature}: p = {p_val:.4e}")

print("\nGenerating UMAP Plot...")
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=df['cluster'], palette='tab20', legend=None)
plt.title('UMAP Projection with Noise Highlighted')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.grid(True)
plt.show()

print("\nRunning Outlier Detection inside Noise Points...")
X_noise = df_noise[features].fillna(0)
X_noise_scaled = scaler.transform(X_noise)

iso = IsolationForest(contamination=0.1, random_state=42)
iso_outliers = iso.fit_predict(X_noise_scaled)

lof = LocalOutlierFactor(n_neighbors=20)
lof_outliers = lof.fit_predict(X_noise_scaled)

svm = OneClassSVM(kernel='rbf', nu=0.1)
svm_outliers = svm.fit_predict(X_noise_scaled)

print(f"Isolation Forest flagged {np.sum(iso_outliers == -1)} noise points as outliers.")
print(f"Local Outlier Factor flagged {np.sum(lof_outliers == -1)} noise points as outliers.")
print(f"One-Class SVM flagged {np.sum(svm_outliers == -1)} noise points as outliers.")

pca = PCA(n_components=2, random_state=42)
X_noise_pca = pca.fit_transform(X_noise_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_noise_pca[:, 0], X_noise_pca[:, 1], c=(iso_outliers == -1), cmap='coolwarm', edgecolor='k')
plt.title('PCA of Noise Points - Isolation Forest Outliers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

print("\nRunning Unsupervised Model: Learning Normal Structure from Clustered Points...")
X_clustered = df_clustered[features].fillna(0)
X_clustered_scaled = scaler.transform(X_clustered)

unsupervised_model = IsolationForest(contamination=0.1, random_state=42)
unsupervised_model.fit(X_clustered_scaled)
noise_scores = unsupervised_model.decision_function(X_noise_scaled)

print("\nNoise Points Outlier Scores (Higher = more normal, Lower = more anomalous):")
print(noise_scores)

plt.figure(figsize=(8, 4))
sns.histplot(noise_scores, kde=True, bins=30)
plt.title('Outlier Scores of Noise Points (Learned from Clustered Patients)')
plt.xlabel('Outlier Score')
plt.ylabel('Count')
plt.grid(True)
plt.show()

most_abnormal_idx = np.argsort(noise_scores)[:10]
print("\nTop 10 Most Abnormal Noise Points (Indices):")
print(df_noise.iloc[most_abnormal_idx])

"""Interpretation of Outlier and Noise Cluster Analysis
Although HDBSCAN labeled 459 data points as "noise," 
a detailed unsupervised investigation suggests these may not be true outliers in a traditional sense. 
First, boxplots of features like BMI and Diabetes show nearly identical distributions between noise and clustered points 
— indicating no stark statistical separation. 
This is echoed by the p-values: while certain features (e.g., Stroke, TBI, Diabetes) show extremely significant differences (p < 1e-25), 
others like BMI, MoCA, UPDRS, and even Diagnosis are not significantly different, 
implying that the noise group isn't globally abnormal across all dimensions. 
The UMAP projection visually reinforces this: noise points (blue) are densely packed and overlap heavily with clustered patients, 
and in fact, the entire left-side cluster seems structurally consistent, 
not noisy — possibly reflecting a distinct but legitimate subgroup, rather than outliers. 
The PCA plot further confirms this by showing that Isolation Forest outliers are scattered inside clusters rather than at the edges
 — atypical for true anomalies. 
Finally, the outlier scores learned from clustered points follow a near-normal distribution,
which implies that noise points are not exceptionally "anomalous" when compared to the main population.
Altogether, these results strongly suggest that the so-called noise cluster might actually represent a meaningful, 
possibly clinically distinct subgroup, rather than a group of random or faulty data points.

"""
