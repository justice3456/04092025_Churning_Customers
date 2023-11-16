import streamlit as st  # importing streamlit
import numpy as np  # importing numpy
import pickle  # importing pickle
import pandas as pd  # importing pandas
from sklearn.preprocessing import StandardScaler

st.title('Churning Customers in a Telecoms Company')
#
# col1, col2 = st.columns(2)
#
# col1.image(str('fifa-soccer.png'))
# image_column, content_column = st.columns([1, 4])

# Add the image to the first column

# Add content to the second column

OnlineSecurity = st.radio("Has online security:", ('Yes', 'No', 'No internet service'))
if OnlineSecurity == 'Yes':
    OnlineSecurity = 2
if OnlineSecurity == 'No':
    OnlineSecurity = 0
if OnlineSecurity == 'No internet service':
    OnlineSecurity = 1

OnlineBackup = st.radio("Has online backup:", ('Yes', 'No', 'No internet service'))
if OnlineBackup == 'Yes':
    OnlineBackup = 2
if OnlineBackup == 'No':
    OnlineBackup = 0
if OnlineBackup == 'No internet service':
    OnlineBackup = 1

TechSupport = st.radio("Has tech support:", ('Yes', 'No', 'No internet service'))
if TechSupport == 'Yes':
    TechSupport = 2
if TechSupport == 'No':
    TechSupport = 0
if TechSupport == 'No internet service':
    TechSupport = 1


Contract = st.radio("Contract form:", ('Month-to-month', 'One year', 'Two year'))
if Contract == 'Month-to-month':
    Contract = 0
if Contract == 'One year':
    Contract = 1
if Contract == 'Two year':
    Contract = 2

PaperlessBilling = st.radio("Has tech support:", ('Yes', 'No'))
if PaperlessBilling == 'Yes':
    PaperlessBilling = 1
else:
    PaperlessBilling = 0

Tenure = st.slider("Slide to choose a tenure", min_value=0, max_value=100)
MonthlyCharges = st.slider("Slide to choose monthly charge", min_value=0, max_value=100)

Values = [OnlineSecurity, OnlineBackup, TechSupport, Contract, PaperlessBilling, Tenure, MonthlyCharges]
Values = np.array([Values])

columns = ['OnlineSecurity','OnlineBackup','TechSupport','Contract','PaperlessBilling','tenure','MonthlyCharges']

scaler = pickle.load(open('Churning_scaler1.pkl', 'rb'))
model = pickle.load(open('Churning_model.pkl', 'rb'))

scaled_user_inputs = pd.DataFrame(scaler.transform(Values))

Churn = model.predict(scaled_user_inputs)

threshold = 0.05  # Adjust threshold as needed
binary_predictions = (Churn > threshold).astype(int)

yes = 'Yes'
no = 'No'
if st.button('SUBMIT'):
    if binary_predictions == 0:
        st.write(f'Customer Churn?: {yes}')
    else:
        st.write(f'Churn: {no}')
