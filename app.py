import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('D:/Breast Cancer Prediction/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
def main():
    st.title("Breast Cancer Prediction App")
    st.write("""
    This app uses a machine learning model to predict the likelihood of breast cancer
    based on input features.
    """)
    
    # Define input fields for user to enter feature values
    st.header("Input Features")
    texture_mean = st.number_input("Texture Mean", min_value=0.0)
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0)
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0)
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0)
    symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0)
    texture_se = st.number_input("Texture SE", min_value=0.0)
    area_se = st.number_input("Area SE", min_value=0.0)
    smoothness_se = st.number_input("Smoothness SE", min_value=0.0)
    compactness_se = st.number_input("Compactness SE", min_value=0.0)
    concavity_se = st.number_input("Concavity SE", min_value=0.0)
    concave_points_se = st.number_input("Concave Points SE", min_value=0.0)
    symmetry_se = st.number_input("Symmetry SE", min_value=0.0)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0)
    texture_worst = st.number_input("Texture Worst", min_value=0.0)
    area_worst = st.number_input("Area Worst", min_value=0.0)
    smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0)
    compactness_worst = st.number_input("Compactness Worst", min_value=0.0)
    concavity_worst = st.number_input("Concavity Worst", min_value=0.0)
    concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0)
    symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0)

    # Collect all input features into a dataframe
    input_data = pd.DataFrame({
        'texture_mean': [texture_mean],
        'smoothness_mean': [smoothness_mean],
        'compactness_mean': [compactness_mean],
        'concave points_mean': [concave_points_mean],
        'symmetry_mean': [symmetry_mean],
        'fractal_dimension_mean': [fractal_dimension_mean],
        'texture_se': [texture_se],
        'area_se': [area_se],
        'smoothness_se': [smoothness_se],
        'compactness_se': [compactness_se],
        'concavity_se': [concavity_se],
        'concave points_se': [concave_points_se],
        'symmetry_se': [symmetry_se],
        'fractal_dimension_se': [fractal_dimension_se],
        'texture_worst': [texture_worst],
        'area_worst': [area_worst],
        'smoothness_worst': [smoothness_worst],
        'compactness_worst': [compactness_worst],
        'concavity_worst': [concavity_worst],
        'concave points_worst': [concave_points_worst],
        'symmetry_worst': [symmetry_worst],
        'fractal_dimension_worst': [fractal_dimension_worst]
    })

    # Drop 'Unnamed: 32' if present in the dataset
    if 'Unnamed: 32' in input_data.columns:
        input_data = input_data.drop(columns=['Unnamed: 32'])
        
    # Predict the probability of breast cancer
    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]
        st.write(f"Prediction: {'Malignant (Cancerous)' if prediction[0] == 1 else 'Benign (Not Cancerous)'}")
        st.write(f"Prediction Probability: {prediction_proba:.2f}")

if __name__ == '__main__':
    main()
