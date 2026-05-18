import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="BigMart Sales Prediction",
    page_icon="📈",
    layout="centered"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():

    # Dataset file
    df = pd.read_csv("Train.csv")

    return df


# ---------------- BUILD MODEL ----------------
@st.cache_resource
def build_model(df):

    data = df.copy()

    # Drop ID column
    if "Item_Identifier" in data.columns:
        data.drop(columns=["Item_Identifier"], inplace=True)

    # Target column
    target_column = "Item_Outlet_Sales"

    # Remove rows with missing target
    data = data.dropna(subset=[target_column])

    # Features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    encoders = {}

    # ---------------- HANDLE MISSING VALUES ----------------
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

            X[col] = X[col].fillna(
                X[col].median()
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
        X.columns
    )


# ---------------- MAIN APP ----------------
def main():

    st.title("📈 BigMart Sales Prediction")

    st.write(
        "Predict Item Outlet Sales using KNN Regressor"
    )

    # Load data
    df = load_data()

    st.subheader("📂 Dataset Preview")

    st.dataframe(df.head())

    # Build model
    (
        model,
        scaler,
        encoders,
        mae,
        r2,
        feature_columns
    ) = build_model(df)

    # ---------------- MODEL METRICS ----------------
    st.subheader("📊 Model Performance")

    st.success(f"Mean Absolute Error: {mae:.2f}")

    st.success(f"R² Score: {r2:.2f}")

    # ---------------- USER INPUTS ----------------
    st.subheader("🧾 Enter Product Details")

    item_weight = st.number_input(
        "Item Weight",
        min_value=0.0
    )

    item_fat_content = st.selectbox(
        "Item Fat Content",
        list(df["Item_Fat_Content"].dropna().unique())
    )

    item_visibility = st.number_input(
        "Item Visibility",
        min_value=0.0
    )

    item_type = st.selectbox(
        "Item Type",
        list(df["Item_Type"].dropna().unique())
    )

    item_mrp = st.number_input(
        "Item MRP",
        min_value=0.0
    )

    outlet_identifier = st.selectbox(
        "Outlet Identifier",
        list(df["Outlet_Identifier"].dropna().unique())
    )

    outlet_establishment_year = st.number_input(
        "Outlet Establishment Year",
        min_value=1980
    )

    outlet_size = st.selectbox(
        "Outlet Size",
        list(df["Outlet_Size"].dropna().unique())
    )

    outlet_location_type = st.selectbox(
        "Outlet Location Type",
        list(df["Outlet_Location_Type"].dropna().unique())
    )

    outlet_type = st.selectbox(
        "Outlet Type",
        list(df["Outlet_Type"].dropna().unique())
    )

    # ---------------- PREDICTION ----------------
    if st.button("Predict Sales"):

        input_data = pd.DataFrame([{
            "Item_Weight": item_weight,
            "Item_Fat_Content": item_fat_content,
            "Item_Visibility": item_visibility,
            "Item_Type": item_type,
            "Item_MRP": item_mrp,
            "Outlet_Identifier": outlet_identifier,
            "Outlet_Establishment_Year": outlet_establishment_year,
            "Outlet_Size": outlet_size,
            "Outlet_Location_Type": outlet_location_type,
            "Outlet_Type": outlet_type
        }])

        # Encode categorical columns
        for col in input_data.columns:

            if col in encoders:

                value = str(input_data[col].iloc[0])

                if value not in encoders[col].classes_:
                    value = encoders[col].classes_[0]

                input_data[col] = encoders[col].transform([value])

        # Convert all columns numeric
        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Fill NaN values
        input_data = input_data.fillna(0)

        # Match training column order
        input_data = input_data[feature_columns]

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        st.subheader("✅ Prediction")

        st.success(
            f"Predicted Sales: ₹ {prediction:.2f}"
        )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    main()