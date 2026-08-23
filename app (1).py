import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 25px;
}
.result-real {
    padding: 20px;
    border-radius: 12px;
    background-color: #e8f5e9;
    border-left: 6px solid #2e7d32;
    font-size: 24px;
    font-weight: 700;
}
.result-fake {
    padding: 20px;
    border-radius: 12px;
    background-color: #ffebee;
    border-left: 6px solid #c62828;
    font-size: 24px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 Fake Review Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine Learning based review classification using TF-IDF + Logistic Regression</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Fake Review Dataset CSV",
    type=["csv"]
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

max_features = st.sidebar.slider(
    "TF-IDF Max Features",
    min_value=1000,
    max_value=20000,
    value=10000,
    step=1000
)

# -----------------------------
# Load Dataset
# -----------------------------
if uploaded_file is None:
    st.info(
        "👈 Please upload your **fake reviews dataset CSV** from the sidebar."
    )
    st.markdown("### Expected columns")
    st.code("category, rating, label, text_")
    st.markdown("""
    **Label meaning in the project:**
    - `CG` = Computer Generated review
    - `OR` = Original review
    """)
    st.stop()

@st.cache_data
def load_data(file):
    data = pd.read_csv(
        file,
        on_bad_lines="skip",
        encoding="latin1"
    )
    return data

df = load_data(uploaded_file)

# -----------------------------
# Dataset validation
# -----------------------------
required_columns = ["category", "rating", "label", "text_"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(
        f"❌ Missing required columns: {', '.join(missing_columns)}"
    )
    st.write("Columns found:", list(df.columns))
    st.stop()

# Clean missing values
df = df.dropna(subset=["text_", "label"]).copy()
df["text_"] = df["text_"].astype(str)
df["label"] = df["label"].astype(str).str.strip()

# Keep the two expected classes if present
valid_labels = ["CG", "OR"]
df = df[df["label"].isin(valid_labels)].copy()

if df.empty:
    st.error("No valid CG/OR labels were found in the dataset.")
    st.stop()

# -----------------------------
# Dataset Overview
# -----------------------------
st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Reviews", f"{len(df):,}")
c2.metric("Categories", df["category"].nunique())
c3.metric("Original Reviews", int((df["label"] == "OR").sum()))
c4.metric("Generated Reviews", int((df["label"] == "CG").sum()))

with st.expander("👀 View Dataset"):
    st.dataframe(df.head(100), use_container_width=True)

# -----------------------------
# Charts
# -----------------------------
st.header("📈 Data Visualization")

tab1, tab2, tab3, tab4 = st.tabs([
    "Pie Chart",
    "Bar Chart",
    "Rating Distribution",
    "Heatmap"
])

with tab1:
    label_counts = df["label"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(
        label_counts.values,
        labels=label_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Fake/Generated vs Original Reviews")
    st.pyplot(fig)

with tab2:
    category_counts = df["category"].value_counts().head(12)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=category_counts.values,
        y=category_counts.index,
        ax=ax
    )
    ax.set_title("Top Review Categories")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Category")
    st.pyplot(fig)

with tab3:
    rating_numeric = pd.to_numeric(df["rating"], errors="coerce")
    rating_df = df.copy()
    rating_df["rating_numeric"] = rating_numeric
    rating_df = rating_df.dropna(subset=["rating_numeric"])

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=rating_df,
        x="rating_numeric",
        hue="label",
        discrete=True,
        multiple="dodge",
        ax=ax
    )
    ax.set_title("Rating Distribution by Review Label")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

with tab4:
    rating_df = df.copy()
    rating_df["rating_numeric"] = pd.to_numeric(
        rating_df["rating"], errors="coerce"
    )

    cross_table = pd.crosstab(
        rating_df["rating_numeric"],
        rating_df["label"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cross_table,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_title("Rating vs Review Label")
    ax.set_xlabel("Review Label")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

# -----------------------------
# Model Training
# -----------------------------
st.header("🤖 Machine Learning Model")

@st.cache_resource
def train_model(texts, labels, test_size_value, features):
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size_value,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=features
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    return (
        vectorizer,
        model,
        X_test_tfidf,
        y_test,
        y_pred,
        accuracy
    )

with st.spinner("Training model... Please wait ⏳"):
    (
        tfidf,
        model,
        X_test_tfidf,
        y_test,
        y_pred,
        accuracy
    ) = train_model(
        df["text_"],
        df["label"],
        test_size,
        max_features
    )

st.success("✅ Model trained successfully!")

m1, m2 = st.columns(2)
m1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
m2.metric("Training Method", "TF-IDF + Logistic Regression")

# -----------------------------
# Classification Report
# -----------------------------
st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(3), use_container_width=True)

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("🎯 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
    ax=ax
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
st.pyplot(fig)

# -----------------------------
# Single Review Prediction
# -----------------------------
st.header("🔮 Predict a New Review")

review_text = st.text_area(
    "Enter a product review:",
    height=180,
    placeholder="Example: This product is excellent. The quality is very good and I would recommend it."
)

if st.button("🔍 Predict Review", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        review_vector = tfidf.transform([review_text])
        prediction = model.predict(review_vector)[0]

        # Probability if supported
        probabilities = model.predict_proba(review_vector)[0]
        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )
        confidence = max(probabilities) * 100

        if prediction == "CG":
            st.markdown(
                '<div class="result-fake">⚠️ Prediction: COMPUTER GENERATED / FAKE REVIEW</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-real">✅ Prediction: ORIGINAL / REAL REVIEW</div>',
                unsafe_allow_html=True
            )

        st.write(f"**Confidence:** {confidence:.2f}%")

        prob_df = pd.DataFrame({
            "Label": list(class_probabilities.keys()),
            "Probability": [
                value * 100 for value in class_probabilities.values()
            ]
        })

        st.bar_chart(
            prob_df.set_index("Label")["Probability"]
        )

        st.info(
            "Note: This prediction is based on patterns learned from the uploaded dataset. "
            "It should be treated as a classification result, not as proof that a review is actually fake."
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Fake Review Detection | BCA Mini Project | TF-IDF + Logistic Regression"
)
