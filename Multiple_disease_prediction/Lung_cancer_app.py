# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:28:25 2024

@author: USER
"""

import numpy as np
import streamlit as st
from warnings import simplefilter
import pickle

simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Upload file
data = pd.read_csv("C:/Users/USER/Multiple_disease_prediction/survey lung cancer.csv")

# replacing Target values with 0 and 1 instead of YES and NO.
data['LUNG_CANCER'].replace('YES',1,inplace=True)
data['LUNG_CANCER'].replace('NO',0,inplace=True)

# replacing Gender values with 0 and 1 instead of male(M) and female(F).
data['GENDER'].replace('F',1,inplace=True)
data['GENDER'].replace('M',0,inplace=True)

X= data.drop(columns='LUNG_CANCER', axis=1)
Y = data['LUNG_CANCER']


# standardizing data 
scaler = StandardScaler()
scaler.fit_transform(X)


#loading the saved model
loaded_model = pickle.load(open('C:/Users/USER/Multiple_disease_prediction/trained_model_Lung_cancer.sav','rb'))


# creating a function for prediction

def lung_cancer_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)

    if (prediction[0] == 0):
      return 'No possibility of Lung cancer'
    else:
      return 'hign possibility of Lung cancer'


def main():
    #st.title("Heart disease predictions ")

    html_temp = '''<div style ="background-color:Purple;padding:13px">
    <h1 style ="color:white;text-align:center;"> Lung Cancer Prediction using ML </h1>
    </div>'''
    
    st.markdown(html_temp, unsafe_allow_html = True)
    
    GENDER=st.selectbox("Choose Your Gender",['Male','Female'])
    AGE=st.number_input("Enter Your Age",0,100,1)
    SMOKING=st.selectbox("Do you Smoke",['Yes','No'])
    YELLOW_FINGERS=st.selectbox("Do you have Yellow-Fingers",['Yes','No'])
    ANXIETY=st.selectbox("Do you have Anxiety",['Yes','No'])
    PEER_PRESSURE=st.selectbox("Do you have Peer-Pressure",['Yes','No'])
    CHRONIC_DISEASE=st.selectbox("Do you have Chronic Disease",['Yes','No'])
    FATIGUE=st.selectbox("Do you have Fatigue",['Yes','No'])
    ALLERGY=st.selectbox("Do you have Allergy",['Yes','No'])
    WHEEZING=st.selectbox("Do you have Wheezing",['Yes','No'])
    ALCOHOL_CONSUMING=st.selectbox("Do you consume Alcohol",['Yes','No'])
    COUGHING=st.selectbox("Do you have Coughing Problem",['Yes','No'])
    SHORTNESS_OF_BREATH=st.selectbox("Do you have Shortness of Breath",['Yes','No'])
    SWALLOWING_DIFFICULTY=st.selectbox("Do you have Swallowing Difficulty",['Yes','No'])
    CHEST_PAIN=st.selectbox("Do you have Chest Pain",['Yes','No'])

    
    
    GENDER = 0 if(GENDER=='Male') else 1
    SMOKING = 2 if(SMOKING=='Yes') else 1
    YELLOW_FINGERS = 2 if(YELLOW_FINGERS=='Yes') else 1
    ANXIETY = 2 if(ANXIETY=='Yes') else 1
    PEER_PRESSURE = 2 if(PEER_PRESSURE=='Male') else 1
    CHRONIC_DISEASE = 2 if(CHRONIC_DISEASE=='Yes') else 1
    FATIGUE = 2 if(FATIGUE=='Yes') else 1
    ALLERGY = 2 if(ALLERGY=='Yes') else 1
    WHEEZING = 2 if(WHEEZING=='Yes') else 1
    ALCOHOL_CONSUMING = 2 if(ALCOHOL_CONSUMING=='Yes') else 1
    COUGHING = 2 if(COUGHING=='Yes') else 1
    SHORTNESS_OF_BREATH = 2 if(SHORTNESS_OF_BREATH=='Yes') else 1
    SWALLOWING_DIFFICULTY = 2 if(SWALLOWING_DIFFICULTY=='Yes') else 1
    CHEST_PAIN = 2 if(CHEST_PAIN=='Yes') else 1

             
    # code for prediction
    diagnosis = ''
         
    # creating a button for prediction
    if st.button('Predict'):
        diagnosis = lung_cancer_prediction([GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE ,
                                        ALLERGY ,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,
                                        SWALLOWING_DIFFICULTY,CHEST_PAIN])
          
          
    st.success(diagnosis)


if __name__=='__main__':
     main()