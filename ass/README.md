# Unsupervised Learning on Amazon Products Dataset

This is an unsupervised learning application using Streamlit for clustering analysis on the Amazon Products dataset.

## Features

- **Multiple Clustering Algorithms:**
  - K-Means Clustering
  - DBSCAN (Density-based clustering)
  - Hierarchical Clustering

- **Data Preprocessing:**
  - Automatic handling of numeric features with scaling
  - Categorical encoding for text features
  - TF-IDF vectorization for text data

- **Evaluation Metrics:**
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index

- **Visualization:**
  - 2D cluster visualization using PCA
  - Cluster distribution charts
  - Summary statistics by cluster

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Workflow

1. **Load Data:** Upload or specify a CSV file path (Amazon-Products.csv)
2. **Select Features:** Choose which columns to use for clustering
3. **Choose Algorithm:** Select K-Means, DBSCAN, or Hierarchical Clustering
4. **Set Parameters:** Configure algorithm-specific parameters (number of clusters, epsilon, etc.)
5. **Run Clustering:** Click the button to perform clustering
6. **Analyze Results:** View metrics, visualizations, and cluster summaries

## Dataset

Expected to use the Amazon Products dataset with columns like:
- ProductName
- Category
- Discounted_Price
- Original_Price
- Discount_Percentage
- Number_of_Reviews
- etc.

## Notes

- The application automatically detects numeric and categorical features
- Scaling is applied to numeric features for fair clustering
- PCA is used for 2D visualization regardless of the actual feature dimensionality
