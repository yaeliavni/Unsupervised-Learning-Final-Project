# README - Parkinson's Disease Unsupervised Learning Project

## Project Overview

This project uses **unsupervised** and **self-supervised** learning techniques to explore a real-world Parkinson’s Disease (PD) clinical dataset.  
The primary goal is to uncover hidden patient subtypes and clusters without relying on predefined diagnosis labels.

The workflow includes:
- **UMAP** for dimensionality reduction
- **HDBSCAN** for clustering based on density
- **Self-supervised Random Forest feature selection** (only once)
- **Mutual Information (MI) and statistical validation** to assess feature relevance
- **Outlier detection** across the full dataset and within the noise cluster identified by HDBSCAN

---

## Scripts Overview

### 1. `full_analysis_clustering.py`
**Pipeline:**  
- First, clusters patients using all available features (including Stroke) via UMAP + HDBSCAN.
- Then, applies a **self-supervised Random Forest** to select the top 10 features (based on permutation importance).
- Performs a second clustering round (UMAP + HDBSCAN) using the selected features.
- Calculates silhouette scores, mutual information, and generates cluster profile summaries.
- Evaluates cluster stability via bootstrap resampling.

### 2. `RF_selected_clustering.py`
**Pipeline:**  
- Clusters patients using only the Random Forest-selected features.
- Repeats UMAP + HDBSCAN, silhouette evaluation, mutual information analysis, and statistical testing.
- Confirms the validity and stability of the feature selection process.

### 3. `selected_features_clustering.py`
**Pipeline:**  
- Further refines features by keeping only those that were statistically meaningful (p < 0.05) in previous steps.
- Runs final UMAP + HDBSCAN clustering.
- Conducts in-depth statistical testing.
- Visualizes feature distributions across PD and non-PD patients for key features (e.g., Sleep Quality, Tremor, MoCA, Diabetes, Postural Instability).

### 4. `outlier_analysis.py`
**Pipeline:**  
- Runs global outlier detection using Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM (OCSVM).
- Separately investigates the **HDBSCAN noise cluster** using the same techniques, along with PCA and UMAP visualizations.
- Findings suggest most noise points are not errors but clinically meaningful edge cases.

---

## Unique Methodological Highlights

- **Self-supervised Random Forest:**  
  A pseudo-target (row means) was used to select features without using diagnosis labels, preserving unsupervised integrity.

- **Feature Selection Caution:**  
  Diagnosis and Smoking were excluded from final clustering, despite their importance, to improve clinical interpretability.

- **Noise Cluster Handling:**  
  Noise points were retained and analyzed — shown to likely represent real subgroups rather than data entry errors.

- **Bootstrap Validation:**  
  Cluster stability was checked with multiple resamplings, confirming robustness of the clustering results.

---

## Requirements

- Python ≥ 3.7
- Install the following packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn hdbscan scipy
```

---

## How to Run

1. Set the correct `DATA_PATH` inside each script to point to your `parkinsons_disease_data.csv`.
2. Execute the desired script:
```bash
python full_analysis_clustering.py
```
3. Plots and summary tables will appear or be saved to the `Results/` folder.

---

## Key Points to Emphasize

- **The clustering workflow is fully unsupervised**, except for a single feature selection step.
- **Self-supervised Random Forest** is used only once, in a label-free fashion.
- **Rigorous validation** using both statistical hypothesis tests and mutual information scoring.
- **Noise cluster analysis** reveals new potential patient subtypes.

---

## Conclusion

This project shows how unsupervised and self-supervised learning can uncover interpretable and clinically meaningful subgroups in a complex medical dataset.  
The pipeline emphasizes **reproducibility**, **clinical relevance**, and **methodological robustness** through multiple validation steps.

---

## 📎 Link to Full Report / Visualizations (Overleaf)

[View full project report (Overleaf)](https://www.overleaf.com)  
*(Replace with your actual Overleaf link if needed)*

---
