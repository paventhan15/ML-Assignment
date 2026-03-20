import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


@st.cache_data(show_spinner=False)
def load_data(path: str):
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error(f"File not found at: {path}")
        return None


def build_pipeline(df: pd.DataFrame, feature_cols, target_col):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    text_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in text_cols if df[c].nunique() < 40]
    free_text_cols = [c for c in text_cols if c not in cat_cols]

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    if free_text_cols:
        # Use only the first free text column (to avoid huge feature spaces)
        transformers.append(
            ("text", TfidfVectorizer(max_features=2000, stop_words="english"), free_text_cols[0])
        )

    if not transformers:
        raise ValueError("No supported feature columns were detected.")

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    model = LogisticRegression(max_iter=1000, solver="saga", random_state=42)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline, X, y


def main():
    st.set_page_config(page_title="Streamlit Logistic Regression Demo", layout="wide")
    st.title("📈 Logistic Regression Demo (Classification)")

    st.sidebar.header("Data Source")
    default_path = "/kaggle/input/amazon-products-dataset/Amazon-Products.csv"
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
    st.sidebar.markdown("### Target & Features")
    target_col = st.sidebar.selectbox("Target column", df.columns)

    feature_cols = st.sidebar.multiselect(
        "Feature columns",
        options=[c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col][:3],
    )

    if not feature_cols:
        st.warning("Select at least one feature column.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Logistic Regression Settings")
    C = st.sidebar.number_input("Inverse regularization strength (C)", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
    max_iter = st.sidebar.number_input("Max iterations", min_value=100, max_value=5000, value=1000, step=100)
    test_size = st.sidebar.slider("Test set fraction", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
    random_state = st.sidebar.number_input("Random seed", value=42, step=1)

    if st.sidebar.button("Train Logistic Regression"):
        with st.spinner("Training model..."):
            try:
                pipeline, X, y = build_pipeline(df, feature_cols, target_col)
                pipeline.set_params(model__C=float(C), model__max_iter=int(max_iter))

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state)
                )

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                st.success("Training complete")

                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                st.metric("Accuracy", f"{acc:.3f}")
                st.metric("Precision (weighted)", f"{precision:.3f}")
                st.metric("Recall (weighted)", f"{recall:.3f}")
                st.metric("F1 Score (weighted)", f"{f1:.3f}")

                st.subheader("Confusion matrix")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)

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
