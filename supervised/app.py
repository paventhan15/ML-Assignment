import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def find_default_csv() -> str | None:
    """Look for a local copy of the dataset in the current working directory."""
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


@st.cache_data(show_spinner=False)
def load_data(path: str):
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error(f"File not found at: {path}")
        st.info(
            "If you have the CSV locally, put it in this folder or enter its full path in the sidebar."
        )
        return None


def infer_problem_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        # treat as regression if too many unique values
        return "regression" if series.nunique() > 20 else "classification"
    return "classification"


def build_pipeline(df: pd.DataFrame, feature_cols, target_col, problem_type: str):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    text_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in text_cols if df[c].nunique() < 40]
    free_text_cols = [c for c in text_cols if c not in cat_cols]

    preprocess_steps = []
    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numericCols := numeric_cols)
        )
    if cat_cols:
        transformers.append(
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        )
    if free_text_cols:
        transformers.append(
            ("text", TfidfVectorizer(max_features=5000, stop_words="english"), free_text_cols[0])
        )

    if not transformers:
        raise ValueError("No supported feature columns were detected.")

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline, X, y


def main():
    st.set_page_config(page_title="Streamlit Supervised Demo", layout="wide")
    st.title("📊 Simple Streamlit Supervised Learning Demo")

    st.sidebar.header("Data Source")
    default_path = find_default_csv() or "/kaggle/input/amazon-products-dataset/Amazon-Products.csv"
    path = st.sidebar.text_input("CSV file path", value=default_path)
    use_uploader = st.sidebar.checkbox("Upload CSV instead", value=False)

    df = None
    if use_uploader:
        uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, low_memory=False)
    else:
        df = load_data(path)

    if df is None:
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Preview")
    st.sidebar.write(df.head())

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Select Target")
    target_col = st.sidebar.selectbox("Target column", df.columns)

    problem_type = infer_problem_type(df[target_col])
    forced_type = st.sidebar.radio("Problem type", options=["auto", "classification", "regression"], index=0)
    if forced_type != "auto":
        problem_type = forced_type

    st.sidebar.markdown(f"**Detected problem type:** {problem_type}")

    feature_cols = st.sidebar.multiselect(
        "Feature columns",
        options=[c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col][:3],
    )

    if not feature_cols:
        st.warning("Select at least one feature column.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Train / Test Split")
    test_size = st.sidebar.slider("Test set fraction", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
    random_state = st.sidebar.number_input("Random seed", value=42, step=1)

    if st.sidebar.button("Run training"):
        with st.spinner("Training model..."):
            try:
                pipeline, X, y = build_pipeline(df, feature_cols, target_col, problem_type)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state)
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                st.success("Training complete")

                if problem_type == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{acc:.3f}")
                    st.subheader("Sample predictions")
                    results = X_test.copy()
                    results["truth"] = y_test.values
                    results["prediction"] = y_pred
                    st.dataframe(results.head(20))
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = mse ** 0.5
                    r2 = r2_score(y_test, y_pred)
                    st.metric("RMSE", f"{rmse:.3f}")
                    st.metric("R²", f"{r2:.3f}")
                    st.subheader("Sample predictions")
                    results = X_test.copy()
                    results["truth"] = y_test.values
                    results["prediction"] = y_pred
                    st.dataframe(results.head(20))

            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("---")
    st.write("### Dataset info")
    with st.expander("Show dataset info"):
        st.write(df.info())
        st.write(df.describe(include="all"))


if __name__ == "__main__":
    main()
