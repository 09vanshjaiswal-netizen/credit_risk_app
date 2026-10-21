from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳")
folder = Path(__file__).resolve().parent
try:
    model = joblib.load(folder / "Random_Forest_credit.pk")
    encoder_columns = [
        "Sex",
        "Housing",
        "Saving accounts",
        "Checking account",
    ]
    encoders = {
        col: joblib.load(folder / f"{col}_encoder.pkl")
        for col in encoder_columns
    }
    target_encoder = joblib.load(folder / "target_encoder.pkl")
except Exception as error:
    st.title("Credit Risk Prediction App")
    st.error(f"Could not load a model or encoder file: {error}")
    st.write("Check that the files are in the same folder as `aap.py`:")
    st.code(
        "Random_Forest_credit_model.pkl\n"
        "Sex_encoder.pkl\n"
        "Housing_encoder.pkl\n"
        "Saving accounts_encoder.pkl\n"
        "Checking account_encoder.pkl\n"
        "target_encoder.pkl"
    )
    st.stop()
st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict credit risk.")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", list(encoders["Sex"].classes_))
job = st.selectbox("Job", [0, 1, 2, 3])
housing = st.selectbox("Housing", list(encoders["Housing"].classes_))
saving_account = st.selectbox(
    "Saving account",
    list(encoders["Saving accounts"].classes_),
)
checking_account = st.selectbox(
    "Checking account",
    list(encoders["Checking account"].classes_),
)
credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)
if st.button("Predict risk"):
    try:
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": encoders["Sex"].transform([sex])[0],
            "Job": job,
            "Housing": encoders["Housing"].transform([housing])[0],
            "Saving accounts": encoders["Saving accounts"].transform(
                [saving_account]
            )[0],
            "Checking account": encoders["Checking account"].transform(
                [checking_account]
            )[0],
            "Credit amount": credit_amount,
            "Duration": duration,
        }])

        if hasattr(model, "feature_names_in_"):
            input_df = input_df[list(model.feature_names_in_)]

        prediction = model.predict(input_df)[0]
        risk_label = target_encoder.inverse_transform([prediction])[0]
        risk_label = str(risk_label).strip().lower()

        if risk_label == "good":
            st.success("Predicted credit risk: GOOD")
        elif risk_label == "bad":
            st.error("Predicted credit risk: BAD")
        else:
            st.info(f"Predicted credit risk: {risk_label}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")