# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:39:44 2024

@author: USER
"""

import numpy as np
import streamlit as st
import pickle
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Upload file
data = pd.read_csv("C:/Users/USER/Multiple_disease_prediction/parkinsons.csv")
data = data.drop('name', axis=1)
X = data.drop(columns='status',axis=1)
Y = data['status']

# standardize the data
scaler = StandardScaler()
scaler.fit(X)


#loading the saved model
loaded_model = pickle.load(open('C:/Users/USER/Multiple_disease_prediction/trained_model_Parkinson.sav', 'rb'))

# creating a function for prediction

def parkinson_prediction(input_data):
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)


    if (prediction[0] == 0):
        return "The Person does not have Parkinsons Disease"

    else:
        return "The Person has Parkinsons"


def main():
    #st.title("Heart disease predictions ")

    html_temp = '''<div style ="background-color:Purple;padding:13px">
    <h1 style ="color:white;text-align:center;"> Parkinson's Disease Prediction using ML </h1>
    </div>'''
    
    st.markdown(html_temp, unsafe_allow_html = True)


    #st.image(image,caption='Check you have heart disease')

    st.text("Enter Vocal Fundamental Frequencies below")
    
    MDVP_Fo_Hz=st.number_input("Average vocal fundamental frequency MDVP-Fo(Hz)",0.00000,1000.00000)
    MDVP_Fhi_Hz=st.number_input("Maximum vocal fundamental frequency MDVP-Fhi(Hz)",0.00000,1000.00000)
    MDVP_Flo_Hz=st.number_input("Minimum vocal fundamental frequency MDVP-Flo(Hz)",0.00000,1000.00000)
    st.text("Several measures of variation in fundamental frequency")
    MDVP_Jitter_persentage=st.number_input("Enter MDVP-Jitter(persentage)",placeholder="Enter value",min_value=0.00000,max_value=1.00000)
    MDVP_Jitter_Abs=st.number_input("Enter MDVP-Jitter(Abs) value",0.00000,10.00000)
    MDVP_RAP=st.number_input("Enter MDVP-RAP",0.00000,10.00000)
    MDVP_PPQ=st.number_input(" Enter MDVP-PPQ",0.00000,10.00000)
    Jitter_DDP=st.number_input("Enter Jitter-DDP",0.00000,10.00000)
    st.text("Several measures of variation in amplitude")
    MDVP_Shimmer=st.number_input("Enter MDVP-Shimmer",0.00000,10.00000)
    MDVP_Shimmer_dB=st.number_input("Enter MDVP-Shimmer(dB)",0.00000,10.00000)
    Shimmer_APQ3=st.number_input("Enter Shimmer-APQ3",0.00000,10.00000)
    Shimmer_APQ5=st.number_input("Enter Shimmer-APQ5",0.00000,10.00000)
    MDVP_APQ=st.number_input("Enter MDVP-APQ",0.00000,10.00000)
    Shimmer_DDA=st.number_input("Enter Shimmer-DDA",0.00000,10.00000)
    st.text("Two measures of ratio of noise to tonal components in the voice")
    NHR=st.number_input("Enter NHR",0.00000,1000.00000)
    HNR=st.number_input("Enter HNR",0.00000,1000.00000)
    st.text("Two nonlinear dynamical complexity measures")
    RPDE=st.number_input("Enter RPDE",0.000000,100.000000)
    D2=st.number_input("Enter D2",0.000000,100.000000)
    st.text("Signal fractal scaling exponent")
    DFA=st.number_input("Enter DFA",0.000000,100.000000)
    st.text("Three nonlinear measures of fundamental frequency variation")
    spread1=st.number_input("Enter spread1",-100.000000,100.000000)
    spread2=st.number_input("Enter spread2",-100.000000,100.000000)
    PPE=st.number_input("Enter PPE",-100.000000,100.000000)
    
        
    if st.button("Predict"):
        diagnosis=parkinson_prediction([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_persentage,
        MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
        MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5,
        MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
    
        st.success(diagnosis)
    



if __name__=='__main__':
    main()