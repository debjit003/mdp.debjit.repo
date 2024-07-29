# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:20:09 2024

@author: USER
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Upload file
data = pd.read_csv("C:/Users/USER/Multiple_disease_prediction/diabetes.csv")
data = data.drop(columns='Outcome', axis=1)


# standardizing data 
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)


#loading the saved model
loaded_model = pickle.load(open('C:/Users/USER/Multiple_disease_prediction/trained_model_diabetes.sav', 'rb'))

# creating a function for prediction

def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)
    print(std_data)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    
    # giving a title 
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies = st.slider('Number of Pregnancies',0,20)
    Glucose = st.number_input('Glucose Level',0,300,0.001)
    BloodPressure = st.number_input('BloodPressure',0,300,0.1)
    SkinThickness = st.number_input('SkinThickness',0,300,0.1)
    Insulin = st.number_input('Insulin Level',0,1500,0.1)
    BMI = st.slider('BMI value',12.02,94.85)
    DiabetesPedigreeFunction = st.number_input('value of DiabetesPedigreeFunction',0,30,0.001)
    Age = st.number_input('Age of the Person',0,200,1)
    
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin ,BMI ,DiabetesPedigreeFunction ,Age])
        
        
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
    
    