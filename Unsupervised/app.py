import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------------------------
# Find dataset
# ------------------------------
def find_default_csv():
    local_names = ["Amazon-Products.csv", "amazon-products.csv"]
    cwd = Path.cwd()

    for name in local_names:
        candidate = cwd / name
        if candidate.exists():
            return str(candidate)

    for candidate in cwd.glob("Amazon*.csv"):
        if candidate.is_file():
            return str(candidate)

    return None


# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error(f"File not found at: {path}")
        return None


# ------------------------------
# Build preprocessing pipeline
# ------------------------------
def build_pipeline(df, feature_cols, n_clusters):
    X = df[feature_cols].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    text_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in text_cols if df[c].nunique() < 40]
    free_text_cols = [c for c in text_cols if c not in cat_cols]

    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols)
        )

    if cat_cols:
        transformers.append(
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        )

    if free_text_cols:
        transformers.append(
            ("text", TfidfVectorizer(max_features=5000, stop_words="english"), free_text_cols[0])
        )

    preprocessor = ColumnTransformer(transformers)

    model = KMeans(n_clusters=n_clusters, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("cluster", model)
    ])

    return pipeline, X


# ------------------------------
# Main app
# ------------------------------
def main():
    st.set_page_config(page_title="Unsupervised Learning Demo", layout="wide")
    st.title("📊 Unsupervised Learning (Clustering) Demo")

    # Sidebar - Data
    st.sidebar.header("Data Source")
    default_path = find_default_csv() or "Amazon-Products.csv"
    path = st.sidebar.text_input("CSV file path", value=default_path)

    df = load_data(path)

    if df is None:
        st.stop()

    st.sidebar.write("Preview")
    st.sidebar.write(df.head())

    # Feature selection
    st.sidebar.markdown("### Select Features")
    feature_cols = st.sidebar.multiselect(
        "Feature columns",
        options=df.columns,
        default=df.columns[:3]
    )

    if not feature_cols:
        st.warning("Select at least one feature column")
        st.stop()

    # Clustering parameters
    st.sidebar.markdown("### Clustering Settings")
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)

    if st.sidebar.button("Run Clustering"):
        with st.spinner("Clustering data..."):
            try:
                pipeline, X = build_pipeline(df, feature_cols, n_clusters)

                # Fit model
                pipeline.fit(X)

                # Predict clusters
                clusters = pipeline.predict(X)

                df["Cluster"] = clusters

                st.success("Clustering complete!")

                # Show results
                st.subheader("Clustered Data")
                st.dataframe(df.head(20))

                # ------------------------------
                # PCA Visualization
                # ------------------------------
                st.subheader("Cluster Visualization (PCA)")

                preprocessed = pipeline.named_steps["preprocessor"].transform(X)

                pca = PCA(n_components=2)
                reduced = pca.fit_transform(preprocessed.toarray() if hasattr(preprocessed, "toarray") else preprocessed)

                fig, ax = plt.subplots()
                scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters)

                ax.set_xlabel("PCA 1")
                ax.set_ylabel("PCA 2")
                ax.set_title("Cluster Visualization")

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

    # Dataset info
    st.markdown("---")
    st.write("### Dataset Info")
    with st.expander("Show details"):
        st.write(df.info())
        st.write(df.describe(include="all"))


if __name__ == "__main__":
    main()