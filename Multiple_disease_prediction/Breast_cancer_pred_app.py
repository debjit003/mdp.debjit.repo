# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:48:46 2024

@author: USER
"""
import numpy as np
import streamlit as st
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
import pickle



#loading the saved model
loaded_model = pickle.load(open('C:/Users/USER/Multiple_disease_prediction/trained_model_breast_cancer.sav', 'rb'))



def breast_cancer_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    # standardizing data 
    #scaler = StandardScaler()
    #standardized_data = scaler.fit_transform(data)

    # standardize the input data
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    #print(prediction)

    if (prediction[0] == 0):
      return 'The preson is alive without recurrence'
    else:
      return 'Breast Cancer Recurrence or Death'

def main():
    #st.title("Heart disease predictions ")

    html_temp = '''<div style ="background-color:Purple;padding:13px">
    <h1 style ="color:white;text-align:center;"> Breast Cancer Recurrence Prediction using ML </h1>
    </div>'''
    
    st.markdown(html_temp, unsafe_allow_html = True)
    
    age=st.number_input("Age",min_value=0,max_value=100,step=1,placeholder='Enter your age')
    meno=st.selectbox("Menopausal status",['Premenopausal','Postmenopausal'])
    size=st.number_input("Tumor size (in mm)",0.0,100.0,0.1)
    grade=st.selectbox("Tumor grade",[1,2,3])
    nodes=st.slider("Number of positive lymph nodes",1,100,1)
    pgr=st.number_input("Enter progesterone receptors (fmol/l)",0.0,5000.0,0.1)
    er=st.number_input("Enter estrogen receptors (fmol/l)",0.0,5000.0,0.1)
    hormon=st.selectbox("Did Hormonal Therapy",['Yes','No'])
    rfstime=st.number_input("recurrence free survival time: days to first of recurrence, death or last follow-up",0,10000,1)
    
    
    meno = 1 if(meno=='postmenopausal') else 0
    hormon = 1 if(hormon=='Yes') else 0
    
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Predict'):
        diagnosis = breast_cancer_prediction([[age,meno,size,grade,nodes,pgr,er,hormon,
        rfstime]])
        
        
    st.success(diagnosis)

if __name__=='__main__':
    main()
