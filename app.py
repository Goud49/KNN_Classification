import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💼",
    layout="centered"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():

    try:
        df = pd.read_csv("adult.csv")

    except FileNotFoundError:

        st.error(
            "adult.csv file not found. Upload dataset to GitHub repository."
        )

        st.stop()

    return df


# ---------------- BUILD MODEL ----------------
@st.cache_resource
def build_model(df):

    data = df.copy()

    # Replace ? with NaN
    data = data.replace("?", np.nan)

    # Drop missing rows
    data = data.dropna()

    # Target column
    target_column = "income"

    # Features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode target
    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(y)

    encoders = {}

    # ---------------- HANDLE FEATURES ----------------
    for col in X.columns:

        # Categorical columns
        if X[col].dtype == "object":

            encoder = LabelEncoder()

            X[col] = encoder.fit_transform(
                X[col].astype(str)
            )

            encoders[col] = encoder

        # Numeric columns
        else:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            X[col] = X[col].fillna(
                X[col].median()
            )

    # ---------------- SCALE ----------------
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------- MODEL ----------------
    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(X_train, y_train)

    # ---------------- EVALUATION ----------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred
    )

    return (
        model,
        scaler,
        encoders,
        accuracy,
        report,
        X.columns,
        data,
        target_encoder
    )


# ---------------- MAIN APP ----------------
def main():

    st.title("💼 Adult Income Prediction")

    st.write(
        "Predict whether income is <=50K or >50K using KNN Classification."
    )

    # Load dataset
    df = load_data()

    st.subheader("📂 Dataset Preview")

    st.dataframe(df.head())

    # Build model
    (
        model,
        scaler,
        encoders,
        accuracy,
        report,
        feature_columns,
        data,
        target_encoder
    ) = build_model(df)

    # ---------------- METRICS ----------------
    st.subheader("📊 Model Accuracy")

    st.success(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    st.subheader("📝 Classification Report")

    st.text(report)

    # ---------------- USER INPUTS ----------------
    st.subheader("🧾 Enter Person Details")

    input_data_dict = {}

    for col in feature_columns:

        # Categorical
        if data[col].dtype == "object":

            options = list(
                data[col].unique()
            )

            input_data_dict[col] = st.selectbox(
                col,
                options
            )

        # Numeric
        else:

            input_data_dict[col] = st.number_input(
                col,
                value=float(data[col].median())
            )

    # ---------------- PREDICTION ----------------
    if st.button("Predict Income"):

        input_data = pd.DataFrame(
            [input_data_dict]
        )

        # Encode categorical columns
        for col in input_data.columns:

            if col in encoders:

                value = str(
                    input_data[col].iloc[0]
                )

                if value not in encoders[col].classes_:
                    value = encoders[col].classes_[0]

                input_data[col] = encoders[col].transform(
                    [value]
                )

        # Convert numeric
        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        input_data = input_data.fillna(0)

        # Match training columns
        input_data = input_data[
            feature_columns
        ]

        # Scale input
        input_scaled = scaler.transform(
            input_data
        )

        # Prediction
        prediction = model.predict(
            input_scaled
        )[0]

        result = target_encoder.inverse_transform(
            [prediction]
        )[0]

        # ---------------- OUTPUT ----------------
        st.subheader("✅ Prediction")

        st.success(
            f"Predicted Income: {result}"
        )


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
