import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# ---------------- Paths ----------------
working_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------- Load Models ----------------
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
liver_model, liver_scaler = pickle.load(open('saved_models/liver_model.sav','rb'))

# ---------------- Sidebar ----------------
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction',
         'Liver Disease Prediction'],   # ✅ Added here
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'droplet'],  # ✅ icon added
        default_index=0
    )

# =========================================================
# Diabetes Prediction
# =========================================================
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]
        prediction = diabetes_model.predict([user_input])

        diab_diagnosis = 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

    st.success(diab_diagnosis)

# =========================================================
# Heart Disease Prediction
# =========================================================
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1: age = st.text_input('Age')
    with col2: sex = st.text_input('Sex')
    with col3: cp = st.text_input('Chest Pain types')

    with col1: trestbps = st.text_input('Resting Blood Pressure')
    with col2: chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1: restecg = st.text_input('Resting ECG results')
    with col2: thalach = st.text_input('Max Heart Rate achieved')
    with col3: exang = st.text_input('Exercise Induced Angina')

    with col1: oldpeak = st.text_input('ST depression')
    with col2: slope = st.text_input('Slope')
    with col3: ca = st.text_input('Major vessels')

    with col1: thal = st.text_input('thal value')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]
        prediction = heart_disease_model.predict([user_input])

        heart_diagnosis = 'Heart disease detected' if prediction[0] == 1 else 'No heart disease'

    st.success(heart_diagnosis)

# =========================================================
# Parkinson's Prediction
# =========================================================
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction")

    inputs = [st.text_input(f'Feature {i}') for i in range(1, 23)]

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        user_input = [float(x) for x in inputs]
        prediction = parkinsons_model.predict([user_input])

        parkinsons_diagnosis = "Parkinson's detected" if prediction[0] == 1 else "No Parkinson's"

    st.success(parkinsons_diagnosis)

# =========================================================
# Liver Disease Prediction
# =========================================================
if selected == 'Liver Disease Prediction':

    st.title('Liver Disease Prediction')

    Age = st.number_input('Age')
    Gender = st.number_input('Gender (1=Male, 0=Female)')
    TB = st.number_input('Total Bilirubin')
    DB = st.number_input('Direct Bilirubin')
    Alkphos = st.number_input('Alkaline Phosphotase')
    SGPT = st.number_input('Alamine Aminotransferase')
    SGOT = st.number_input('Aspartate Aminotransferase')
    TP = st.number_input('Total Proteins')
    ALB = st.number_input('Albumin')
    AG = st.number_input('Albumin and Globulin Ratio')

    if st.button('Liver Test Result'):
        scaled = liver_scaler.transform([[Age, Gender, TB, DB, Alkphos,
                                  SGPT, SGOT, TP, ALB, AG]])

        prediction = liver_model.predict(scaled)

        if prediction[0] == 1:
            st.success('Liver Disease Detected')
        else:
            st.success('No Liver Disease')
