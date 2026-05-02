from __future__ import annotations

import streamlit as st
from streamlit_option_menu import option_menu

from .config import AppConfig
from .inference import ModelBundle, predict_binary


def parse_float_inputs(values: list[str], feature_group: str) -> list[float]:
    parsed: list[float] = []
    for idx, value in enumerate(values, start=1):
        if value == "":
            raise ValueError(f"Please fill all fields in {feature_group}. Missing value at position {idx}.")
        try:
            parsed.append(float(value))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in {feature_group} at position {idx}: '{value}'.") from exc
    return parsed


def render_header() -> None:
    st.title("🧠 AI Health Assistant")
    st.caption(
        "A multi-model screening app for Diabetes, Heart Disease, Parkinson's, and Liver Disease. "
        "For educational use only — not a clinical diagnosis tool."
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### Navigation")
        selected = option_menu(
            "Risk Screening Modules",
            [
                "Diabetes",
                "Heart Disease",
                "Parkinson's",
                "Liver Disease",
            ],
            menu_icon="hospital-fill",
            icons=["activity", "heart", "person", "droplet"],
            default_index=0,
        )
        st.markdown("---")
        st.info("This tool supports rapid demo screening workflows for analytics portfolios.")
    return selected


def render_diabetes(models: ModelBundle) -> None:
    st.subheader("Diabetes Risk Prediction")
    cols = st.columns(3)
    labels = [
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI",
        "Diabetes Pedigree Function", "Age"
    ]
    values = [cols[i % 3].text_input(label, key=f"diabetes_{i}") for i, label in enumerate(labels)]

    if st.button("Run Diabetes Screening"):
        try:
            features = parse_float_inputs(values, "Diabetes")
            pred = predict_binary(models.diabetes, features)
            st.success("Higher diabetes risk detected." if pred == 1 else "Lower diabetes risk detected.")
        except ValueError as err:
            st.error(str(err))


def render_heart(models: ModelBundle) -> None:
    st.subheader("Heart Disease Risk Prediction")
    cols = st.columns(3)
    labels = [
        "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol", "Fasting Blood Sugar",
        "Resting ECG", "Max Heart Rate", "Exercise Induced Angina", "ST Depression", "Slope", "Major Vessels", "Thal"
    ]
    values = [cols[i % 3].text_input(label, key=f"heart_{i}") for i, label in enumerate(labels)]

    if st.button("Run Heart Screening"):
        try:
            features = parse_float_inputs(values, "Heart Disease")
            pred = predict_binary(models.heart, features)
            st.success("Higher heart disease risk detected." if pred == 1 else "Lower heart disease risk detected.")
        except ValueError as err:
            st.error(str(err))


def render_parkinsons(models: ModelBundle) -> None:
    st.subheader("Parkinson's Risk Prediction")
    values = [st.text_input(f"Feature {i}", key=f"park_{i}") for i in range(1, 23)]

    if st.button("Run Parkinson's Screening"):
        try:
            features = parse_float_inputs(values, "Parkinson's")
            pred = predict_binary(models.parkinsons, features)
            st.success("Higher Parkinson's risk detected." if pred == 1 else "Lower Parkinson's risk detected.")
        except ValueError as err:
            st.error(str(err))


def render_liver(models: ModelBundle) -> None:
    st.subheader("Liver Disease Risk Prediction")
    fields = [
        "Age", "Gender (1=Male, 0=Female)", "Total Bilirubin", "Direct Bilirubin", "Alkaline Phosphotase",
        "Alamine Aminotransferase", "Aspartate Aminotransferase", "Total Proteins", "Albumin", "A/G Ratio"
    ]
    values = [st.number_input(field, value=0.0, key=f"liver_{i}") for i, field in enumerate(fields)]

    if st.button("Run Liver Screening"):
        scaled = models.liver_scaler.transform([values])
        pred = int(models.liver.predict(scaled)[0])
        st.success("Higher liver disease risk detected." if pred == 1 else "Lower liver disease risk detected.")


def run_app(models: ModelBundle) -> None:
    config = AppConfig()
    st.set_page_config(page_title=config.page_title, layout=config.layout, page_icon=config.page_icon)
    render_header()
    selected = render_sidebar()

    if selected == "Diabetes":
        render_diabetes(models)
    elif selected == "Heart Disease":
        render_heart(models)
    elif selected == "Parkinson's":
        render_parkinsons(models)
    else:
        render_liver(models)
