import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model('model.h5')

with open('Ohe_geographpy.pkl', 'rb') as file:
    ohe=pickle.load(file)

with open('Lable_encoder_gender.pkl', 'rb') as file:
    lable_encoder=pickle.load(file)
  
with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)
    
    
    
## steamlit app
st.title('Customer Salary Prediction')
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', lable_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_number = st.selectbox('Is Active Member', [0,1])
exited = st.selectbox('Exited',[0,1])


input_data = {
    "CreditScore" : [credit_score],
    "Gender" : [lable_encoder.transform([gender])[0]],
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_number],
    "Exited" : [exited]
}

# One-hot encode 'Geography'
df = pd.DataFrame(input_data)
geo_encoded = ohe.transform([[geography]])
geo = pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out())

# Combine one-hot encoded columns with input data
df = pd.concat([df.reset_index(drop=True), geo], axis=1)

# Scale the input data
scaled_dataFrame = scalar.transform(df)

# Predict churn
prediction = model.predict(scaled_dataFrame)
prediction_proba = prediction[0][0]


st.write(f"The salary of the customer is {prediction_proba:.2f}")
