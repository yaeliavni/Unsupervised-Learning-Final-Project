import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal
import umap
import warnings
import os

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

DATA_PATH = "data/parkinsons_disease_data.csv"

def load_and_preprocess(selected_features_only=False, include_stroke=True):
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["PatientID"], errors="ignore")

    df_numeric = df.select_dtypes(include=[np.number])

    if selected_features_only:
        selected_features = ['Age', 'SleepQuality', 'Diabetes', 'Depression', 'UPDRS', 'MoCA', 'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability']
        if include_stroke:
            selected_features.append('Stroke')
        df_numeric = df_numeric[selected_features]
    elif not include_stroke:
        df_numeric = df_numeric.drop(columns=['Stroke'], errors='ignore')

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    if selected_features_only and 'UPDRS' in df_scaled.columns:
        df_scaled['UPDRS'] *= 2.0

    return df, df_scaled

def reduce_dimensions(X, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(X)

def cluster_data(X_reduced):
    model = hdbscan.HDBSCAN(min_cluster_size=10)
    return model.fit_predict(X_reduced)

def evaluate_clustering(X_reduced, labels):
    if len(set(labels)) > 1:
        return silhouette_score(X_reduced, labels)
    else:
        return -1



import seaborn as sns

def evaluate_and_plot(X_reduced, labels, label_suffix):
    score = evaluate_clustering(X_reduced, labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    plt.figure(figsize=(7, 6))

    # Dynamically generate enough colors
    palette = sns.color_palette("hsv", n_colors=len(unique_labels))

    print(f"\n=== Plot Statistics: {label_suffix} ===")
    for idx, label in enumerate(unique_labels):
        mask = (labels == label)
        cluster_data = X_reduced[mask]
        mean_x = np.mean(cluster_data[:, 0])
        mean_y = np.mean(cluster_data[:, 1])
        se_x = np.std(cluster_data[:, 0], ddof=1) / np.sqrt(cluster_data.shape[0])
        se_y = np.std(cluster_data[:, 1], ddof=1) / np.sqrt(cluster_data.shape[0])

        print(f"Cluster {label}: UMAP1 = {mean_x:.2f} ± {se_x:.2f}, UMAP2 = {mean_y:.2f} ± {se_y:.2f}")

        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    s=30, color=palette[idx], label=f'Cluster {label}' if label != -1 else 'Noise')

    plt.title(f'HDBSCAN Clustering ({label_suffix})\nSilhouette={score:.2f}, Clusters={n_clusters}', fontsize=13)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
    plt.tight_layout()

    save_dir = r"C:\Users\avni1\Documents\שנה ג\unsupervised learning\Final Data For Project\UnsupervisedLearningFinalProject"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{label_suffix.replace(' ', '_')}_Clustering.svg"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='svg', dpi=600, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()

    plt.show()

    return labels




def compute_MI_per_cluster(df_scaled, labels):
    print("\n=== Mutual Information Scores per Cluster ===")
    df_scaled = df_scaled.copy()
    df_scaled['Cluster'] = labels

    all_clusters = sorted(df_scaled['Cluster'].unique())
    mi_results = []

    for cluster in all_clusters:
        if cluster == -1:
            continue  # Skip noise
        df_cluster = df_scaled[df_scaled['Cluster'] == cluster].drop(columns=['Cluster'])
        if len(df_cluster) < 5:
            print(f"Skipping cluster {cluster} due to too few samples ({len(df_cluster)})")
            continue
        for feature in df_cluster.columns:
            X_other = df_cluster.drop(columns=[feature])
            y_feature = df_cluster[feature]
            mi = mutual_info_regression(X_other, y_feature, random_state=42)
            mi_score = np.mean(mi)
            significance = "Significant" if mi_score > 0.05 else "Not Significant"
            mi_results.append((cluster, feature, mi_score, significance))

    mi_df = pd.DataFrame(mi_results, columns=["Cluster", "Feature", "MI_Score", "Significance"])
    print(mi_df)
    return mi_df
def bootstrap_stability_check(df_scaled, n_iterations=20):
    print("\n=== Bootstrapping Cluster Stability ===")
    cluster_counts = []

    for i in range(n_iterations):
        X_sample = resample(df_scaled, replace=True, n_samples=int(0.8 * len(df_scaled)))
        X_reduced_sample = reduce_dimensions(X_sample)
        labels_sample = cluster_data(X_reduced_sample)
        n_clusters = len(set(labels_sample)) - (1 if -1 in labels_sample else 0)
        cluster_counts.append(n_clusters)

    plt.figure(figsize=(7, 5))
    sns.histplot(cluster_counts, bins=range(1, max(cluster_counts) + 2), kde=False)
    plt.xlabel('Number of Clusters Found')
    plt.ylabel('Frequency')
    plt.title('Cluster Counts Across Bootstraps')
    plt.show()

    mean_clusters = np.mean(cluster_counts)
    se_clusters = np.std(cluster_counts, ddof=1) / np.sqrt(len(cluster_counts))  # SE = SD / sqrt(n)

    print(f"Clusters found (mean ± SE): {mean_clusters:.2f} ± {se_clusters:.2f}")
    print("Higher consistency = more stable clustering; large variance = less stable.")


def format_p(p):
    if p < 1e-4:
        return "< 1×10⁻⁴"
    else:
        return f"{p:.2e}"

# changing std to standard error in statistical_testing

def statistical_testing(df_original, labels):
    df_original['Cluster'] = labels
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns.drop('Cluster')
    cluster_labels = df_original['Cluster'].unique()

    print("\nCluster Profile Means ± Standard Error:")
    means = df_original.groupby('Cluster')[numeric_cols].mean()
    ses = df_original.groupby('Cluster')[numeric_cols].sem()  # SE = SD / sqrt(n)
    for feature in numeric_cols:
        print(f"\nFeature: {feature}")
        for cluster in means.index:
            mean_val = means.loc[cluster, feature]
            se_val = ses.loc[cluster, feature]
            print(f"  Cluster {cluster}: {mean_val:.3f} ± {se_val:.3f} (SE)")

    print("\nStatistical Tests (T-test/U-test or ANOVA/Kruskal):")
    p_values_summary = []

    for feature in numeric_cols:
        groups = [df_original[df_original['Cluster'] == label][feature] for label in cluster_labels]
        if len(cluster_labels) == 2:
            _, p = ttest_ind(groups[0], groups[1], nan_policy='omit')
        else:
            _, p = f_oneway(*groups)
        p_values_summary.append((feature, p))

    p_values_df = pd.DataFrame(p_values_summary, columns=['Feature', 'P-Value'])
    p_values_df['Significance'] = np.where(p_values_df['P-Value'] < 0.05, 'Significant', 'Not Significant')
    print(p_values_df)
    return p_values_df

def parkinsons_distribution(df_original, labels):
    df_original['Cluster'] = labels
    if 'Diagnosis' not in df_original.columns:
        print("Diagnosis column missing!")
        return

    summary = df_original.groupby('Cluster').agg(
        Total_Patients=('Diagnosis', 'count'),
        PD_Patients=('Diagnosis', 'sum')
    )
    summary['PD_Percentage'] = (summary['PD_Patients'] / summary['Total_Patients']) * 100

    print("\n=== Parkinson's Patient Distribution by Clusters ===")
    print(summary)

def full_analysis(selected_features_only, include_stroke, label_suffix):
    df_original, df_scaled = load_and_preprocess(selected_features_only=selected_features_only, include_stroke=include_stroke)
    X_reduced = reduce_dimensions(df_scaled)
    labels = cluster_data(X_reduced)
    final_labels = evaluate_and_plot(X_reduced, labels, label_suffix)
    p_values_df = statistical_testing(df_original.copy(), final_labels)
    mi_df = compute_MI_per_cluster(df_scaled.copy(), final_labels)
    parkinsons_distribution(df_original.copy(), final_labels)
    bootstrap_stability_check(df_scaled)

    merged_df = mi_df.merge(p_values_df, on='Feature', how='left')
    important_features = merged_df[(merged_df['MI_Score'] > 0.05) & (merged_df['P-Value'] < 0.05)]

    print("\n=== Important Features per Cluster (MI and p-value based) ===")
    print(important_features[['Cluster', 'Feature', 'MI_Score', 'P-Value']])

def main_pipeline():
    print("UMAP and HDBscan")
    full_analysis(selected_features_only=False, include_stroke=True, label_suffix="Full")
    print("\nUMAP and HDBscan for features selected")
    full_analysis(selected_features_only=True, include_stroke=True, label_suffix="Selected Features ")

if __name__ == "__main__":
    main_pipeline()
