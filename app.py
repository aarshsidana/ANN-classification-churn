import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the label encoders and scaler
with open('label_encode_gender.pkl', 'rb') as f:
    label_encode_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields for user data
Geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
Gender = st.selectbox("Gender", label_encode_gender.classes_)
Age= st.slider("Age", 18, 92)
Balance = st.number_input("Balance", min_value=0.0)
CreditScore = st.slider("Credit Score", 350, 850)
estimatedSalary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", 0, 10)
NumOfProducts = st.slider("Number of Products", 1, 4)
Has_cr_Card = st.selectbox("Has Credit Card", [0, 1])
Is_ActiveMember = st.selectbox("Is Active Member", [0, 1])

# Create a dictionary to hold the input data
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [label_encode_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [Has_cr_Card],
    'IsActiveMember': [Is_ActiveMember],
    'EstimatedSalary': [estimatedSalary]
})

#One-hot encode the 'Geography' feature
geo_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Drop the original 'Geography' column and concatenate the one-hot encoded columns

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the output
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

