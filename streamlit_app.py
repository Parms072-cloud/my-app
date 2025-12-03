import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------- LOAD MODEL & ARTIFACTS ---------
@st.cache_resource
def load_model():
    artifacts = joblib.load("service_interval_model.joblib")
    model = artifacts["model"]
    label_encoders = artifacts["label_encoders"]
    feature_columns = artifacts["feature_columns"]
    return model, label_encoders, feature_columns


model, label_encoders, feature_columns = load_model()

# Adjust this list if you encoded different columns
CATEGORICAL_COLS = list(label_encoders.keys())
DATE_DERIVED_COLS = ["service_year", "service_month"]


# --------- STREAMLIT UI ---------
st.title("Vehicle Service Due Date Predictor")
st.write(
    "Enter the vehicle details and the **last service date** to predict "
    "the **next service due date**."
)

# ---- 1. INPUT: LAST SERVICE DATE ----
service_date = st.date_input(
    "Last service date",
    value=dt.date.today(),
    help="The date when the vehicle was last serviced.",
)

# Derive year and month from the selected date
service_year = service_date.year
service_month = service_date.month

# ---- 2. INPUT: CATEGORICAL FEATURES (make, model, dealer etc.) ----
st.subheader("Vehicle Information")

cat_inputs = {}
for col in CATEGORICAL_COLS:
    le = label_encoders[col]
    # Use label encoder classes as options
    options = list(le.classes_)
    default_idx = 0  # you can change this if you want a different default
    cat_inputs[col] = st.selectbox(
        f"{col.capitalize()}",
        options=options,
        index=default_idx if default_idx < len(options) else 0,
    )

# ---- 3. INPUT: OTHER NUMERIC FEATURES ----
st.subheader("Additional Details")

# Numeric features are all feature columns except categorical + date-derived
numeric_features = [
    col
    for col in feature_columns
    if col not in CATEGORICAL_COLS + DATE_DERIVED_COLS
]

numeric_inputs = {}

for col in numeric_features:
    # Simple heuristic: allow negative? no → min_value=0
    numeric_inputs[col] = st.number_input(
        f"{col}",
        value=0.0,
        step=1.0,
        format="%.2f",
    )

# ---- 4. BUILD FEATURE ROW FOR PREDICTION ----
# Initialize a single-row DataFrame with all feature columns
input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = 0  # start with zeros

# Set date-derived features
if "service_year" in feature_columns:
    input_data.loc[0, "service_year"] = service_year
if "service_month" in feature_columns:
    input_data.loc[0, "service_month"] = service_month

# Set categorical features (encode using label encoders)
for col, val in cat_inputs.items():
    le = label_encoders[col]
    encoded_val = le.transform([val])[0]
    input_data.loc[0, col] = encoded_val

# Set numeric features
for col, val in numeric_inputs.items():
    input_data.loc[0, col] = float(val)


# ---- 5. PREDICT & CALCULATE NEXT SERVICE DATE ----
if st.button("Predict Next Service Due Date"):
    # Model predicts interval_in_months
    interval_months = model.predict(input_data)[0]

    # For safety, round to nearest whole month
    interval_months_rounded = int(round(interval_months))

    # Convert months to a date offset
    # (approximate: 1 month ≈ 30 days; or use pandas DateOffset)
    next_service_timestamp = pd.Timestamp(service_date) + pd.DateOffset(
        months=interval_months_rounded
    )
    next_service_date = next_service_timestamp.date()

    st.markdown("---")
    st.subheader("Prediction Result")

    st.write(f"**Estimated interval:** {interval_months:.2f} months")
    st.write(f"**Rounded interval used:** {interval_months_rounded} months")
    st.write(
        f"👉 **Next service due date is likely around:** "
        f"🗓️ **{next_service_date.strftime('%Y-%m-%d')}**"
    )

    # Optional: extra explanation
    st.info(
        "This is an estimate based on historical service data for similar vehicles. "
        "Always follow manufacturer recommendations and dealer advice."
        "By Parminder"
        
    )
