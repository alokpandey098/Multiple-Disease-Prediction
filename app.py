import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

diabetes_model = pickle.load(open('daibetes.pkl', 'rb'))
heart_model = pickle.load(open('Heart.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons.pkl', 'rb'))

with st.sidebar:
    
    selected = option_menu('Multiple Disease Predictor',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Predictor')
    
    
    # getting the input data from the user
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
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Predictor')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=feature_names)
        heart_prediction = heart_model.predict(input_data)
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart issues'
        else:
          heart_diagnosis = 'The person does not have any heart issues'
        
    st.success(heart_diagnosis)

# Parkinsons prediction system

# Define feature names used during model training
feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 
                  'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
                  'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

def convert_to_float(value):
    """
    Convert input value to float, or return NaN if conversion fails.
    """
    try:
        return float(value)
    except ValueError:
        return np.nan

def prepare_input_data(input_values):
    """
    Prepares and cleans input data for model prediction.
    """
    # Create a DataFrame with the input values and feature names
    input_df = pd.DataFrame([input_values], columns=feature_names)
    
    # Convert all values to numeric, coercing errors to NaN
    input_df = input_df.apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values with 0 or another appropriate value
    input_df.fillna(0, inplace=True)
    
    return input_df

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # Page title
    st.title("Parkinson's Disease Predictor")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    # Code for Prediction
    parkinsons_diagnosis = ''
    
    # Create a button for Prediction    
    if st.button("Parkinson's Test Result"):
        # Convert inputs to float
        inputs_data = [
            convert_to_float(fo), convert_to_float(fhi), convert_to_float(flo),
            convert_to_float(Jitter_percent), convert_to_float(Jitter_Abs), convert_to_float(RAP),
            convert_to_float(PPQ), convert_to_float(DDP), convert_to_float(Shimmer),
            convert_to_float(Shimmer_dB), convert_to_float(APQ3), convert_to_float(APQ5),
            convert_to_float(APQ), convert_to_float(DDA), convert_to_float(NHR),
            convert_to_float(HNR), convert_to_float(RPDE), convert_to_float(DFA),
            convert_to_float(spread1), convert_to_float(spread2), convert_to_float(D2),
            convert_to_float(PPE)
        ]

        # Prepare the input data for prediction
        prepared_data = prepare_input_data(inputs_data)
        
        # Predict using the model
        try:
            parkinsons_prediction = parkinsons_model.predict(prepared_data)
            
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
            
            st.success(parkinsons_diagnosis)
        except Exception as e:
            st.error(f'Error occurred during prediction: {e}')

