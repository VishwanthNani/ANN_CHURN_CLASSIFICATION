import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load the pretrained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehotencoder_geo.pkl', 'rb') as f:
    onehotencoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


#Streamlit app
st.title("Customer Churn Prediction")

#User input
st.header("Enter Customer Information")
CreditScore = st.number_input("Credit Score", min_value=0, max_value=850, value=700)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male","Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)


#Prepare the input Data
input_data =pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_credit_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary],
})
    
#One hot encoding for geography
geo_encoded = onehotencoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geo_encoded_df], axis=1)
input_data.drop(['Geography'], axis=1, inplace=True)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#Make prediction
prediction = model.predict(input_data_scaled)
prediction = (prediction > 0.5).astype(int)
#Display the result
if prediction[0][0] == 1:
    st.success("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")
#Display the input data
st.subheader("Input Data")
st.write(input_data)
#Display the prediction
st.subheader("Prediction")
if prediction[0][0] == 1:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")