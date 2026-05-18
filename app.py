import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="KNN Regression Prediction App",
    page_icon="📈",
    layout="centered"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():

    try:
        df = pd.read_csv("adult
        .csv")

    except FileNotFoundError:

        st.error(
            "Train.csv file not found. Upload Train.csv to GitHub repository."
        )

        st.stop()

    return df


# ---------------- BUILD MODEL ----------------
@st.cache_resource
def build_model(df):

    data = df.copy()

    # Drop unnecessary ID columns
    drop_cols = [
        "Item_Identifier",
        "ID",
        "Id"
    ]

    for col in drop_cols:

        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    # ---------------- TARGET COLUMN ----------------
    # Uses last column automatically
    target_column = data.columns[-1]

    # Remove missing target rows
    data = data.dropna(subset=[target_column])

    # Convert target to numeric
    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    # Remove invalid target rows
    data = data.dropna(subset=[target_column])

    # Features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Store encoders
    encoders = {}

    # ---------------- HANDLE FEATURES ----------------
    for col in X.columns:

        # Categorical columns
        if X[col].dtype == "object":

            X[col] = X[col].fillna(
                X[col].mode()[0]
            )

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

            median_value = X[col].median()

            if pd.isna(median_value):
                median_value = 0

            X[col] = X[col].fillna(
                median_value
            )

    # Final cleaning
    X = X.replace(
        [np.inf, -np.inf],
        0
    )

    X = X.fillna(0)

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
    model = KNeighborsRegressor(
        n_neighbors=5
    )

    model.fit(X_train, y_train)

    # ---------------- EVALUATION ----------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    return (
        model,
        scaler,
        encoders,
        mae,
        r2,
        X.columns,
        target_column,
        data
    )


# ---------------- MAIN APP ----------------
def main():

    st.title("📈 KNN Regression Prediction App")

    st.write(
        "Predict values using K-Nearest Neighbors Regression."
    )

    # Load dataset
    df = load_data()

    st.subheader("📂 Dataset Preview")

    st.dataframe(df.head())

    st.subheader("📋 Dataset Columns")

    st.write(df.columns.tolist())

    # Build model
    (
        model,
        scaler,
        encoders,
        mae,
        r2,
        feature_columns,
        target_column,
        data
    ) = build_model(df)

    # ---------------- METRICS ----------------
    st.subheader("📊 Model Performance")

    st.success(
        f"Mean Absolute Error: {mae:.2f}"
    )

    st.success(
        f"R² Score: {r2:.2f}"
    )

    # ---------------- INPUTS ----------------
    st.subheader("🧾 Enter Feature Values")

    input_data_dict = {}

    for col in feature_columns:

        # Categorical columns
        if data[col].dtype == "object":

            options = list(
                data[col].dropna().unique()
            )

            input_data_dict[col] = st.selectbox(
                col,
                options
            )

        # Numeric columns
        else:

            input_data_dict[col] = st.number_input(
                col,
                value=0.0
            )

    # ---------------- PREDICTION ----------------
    if st.button("Predict"):

        input_data = pd.DataFrame(
            [input_data_dict]
        )

        # Encode categorical values
        for col in input_data.columns:

            if col in encoders:

                value = str(
                    input_data[col].iloc[0]
                )

                # Handle unseen values
                if value not in encoders[col].classes_:
                    value = encoders[col].classes_[0]

                input_data[col] = encoders[col].transform(
                    [value]
                )

        # Convert all to numeric
        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Fill missing values
        input_data = input_data.fillna(0)

        # Match training column order
        input_data = input_data[
            feature_columns
        ]

        # Scale input
        input_scaled = scaler.transform(
            input_data
        )

        # Predict
        prediction = model.predict(
            input_scaled
        )[0]

        # ---------------- OUTPUT ----------------
        st.subheader("✅ Prediction")

        st.success(
            f"Predicted {target_column}: {prediction:.2f}"
        )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    main()
