# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:23:14 2024

@author: USER
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

#loading the saved model

#diabetes
diabetes_model = pickle.load(open('Multiple_disease_prediction/trained_model_diabetes.sav', 'rb'))
scaler = pickle.load(open('Multiple_disease_prediction/scaler_diabetes.sav', 'rb'))

#heart disease
heart_model = pickle.load(open('Multiple_disease_prediction/trained_model_heartDisease.sav', 'rb'))

#breast cancer
brst_cancer_model = pickle.load(open('Multiple_disease_prediction/trained_model_breast_cancer.sav', 'rb'))

#lung cancer
lung_cancer_model = pickle.load(open('Multiple_disease_prediction/trained_model_Lung_cancer.sav','rb'))

#parkinsons disease
parkinsons_model = pickle.load(open('Multiple_disease_prediction/trained_model_Parkinson.sav','rb'))



#icons raw format
#diabetes = """<img width="50" height="50" src="https://img.icons8.com/external-others-pike-picture/50/external-sugar-atherosclerosis-vessel-others-pike-picture.png" alt="external-sugar-atherosclerosis-vessel-others-pike-picture"/>"""




# sidebar for navigate

with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Breast Cancer Prediction',
                            'Lung Cancer Prediction',
                            'Parkinsons Disease Prediction'],
                           icons=['activity','heart-pulse','clipboard2-pulse','lungs','person'],
                           default_index=0)
    
# Diabetes prediction Page
if(selected == 'Diabetes Prediction'):
    
    # Upload file
    data = pd.read_csv("Multiple_disease_prediction/diabetes.csv")
    data = data.drop(columns='Outcome', axis=1)


    # standardizing data 
    #scaler = StandardScaler()
    #standardized_data = scaler.fit_transform(data)
    
    # Page Title
    st.title('Diabetes Prediction using ML')
    #st.markdown(diabetes,unsafe_allow_html=True)
    
    # getting the input data from user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.slider('Number of Pregnancies',0,20)
        SkinThickness = st.number_input('SkinThickness',0.0,300.0,step=0.1)
        DiabetesPedigreeFunction = st.number_input('value of DiabetesPedigreeFunction',0.0,30.0,step=0.001,format="%.3f")
        
    with col2:
        Glucose = st.number_input('Glucose Level',0.0,300.0,step=0.1)
        Insulin = st.number_input('Insulin Level',0.0,1500.0,step=0.1)
        Age = st.number_input('Age of the Person',0,200,1)
        
    with col3:
        BloodPressure = st.number_input('BloodPressure',0.0,300.0,step=0.1)
        BMI = st.slider('BMI value',12.02,94.85)
        
    
    #code for prediction
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    # Print debugging information
    st.write("Input Data Shape:", input_data_reshaped.shape)
    st.write("Input Data:", input_data_as_numpy_array)
    
    # Standardize the input data
    try:
        std_data = scaler.transform(input_data_reshaped)
        st.write("Standardized Data:", std_data)
    except ValueError as e:
        st.error(f"Error during scaling: {e}")
    
    # Creating a button for prediction
    if st.button('Predict'):
        try:
            diab_prediction = diabetes_model.predict(std_data)
            
            if diab_prediction[0] == 0:
                st.success('The Person is not Diabetic')
            else:
                st.warning('The Person is Diabetic')
        except Exception as e:
            st.error(f"Prediction error: {e}")
            
# Heart Disease prediction Page
if(selected == 'Heart Disease Prediction'):
    # Page Title
    st.title('Heart Disease Prediction using ML')
    
    #html_temp = '''<div style ="background-color:Purple;padding:13px">
    #<h1 style ="color:white;text-align:center;"> Heart Disease Prediction using ML </h1>
    #</div>'''
    
    #st.markdown(html_temp, unsafe_allow_html = True)
    
    # getting the input data from user
    #BMI, Smoking, AlcoholDrinking, Stroke,PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory,Race, Diabetic, PhysicalActivity, GenHealth, SleepTime,Asthma, KidneyDisease, SkinCancer
    
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        BMI=st.slider("Enter body mass index",12.02,94.85)
        Stroke=st.selectbox("your Stroke status",['Yes','No'])
        DiffWalking=st.selectbox("your DiffWalking status",['Yes','No'])
        Race=st.selectbox("Ethnicity",['White','Black','Asian','American Indian/Alaskan Native','Other','Hispanic'])
        GenHealth=st.selectbox("your GenHealth status",['Excellent','Very good','Fair','Good','Poor'])
        KidneyDisease=st.selectbox("your KidneyDiseas status",['Yes','No'])
        
    with col2:
        Smoking=st.selectbox("your Smoking status",['Yes','No'])
        PhysicalHealth=st.slider("Enter PhysicalHealth",0.0,30.0,0.1)
        Sex=st.selectbox("Gender",['Male','Female'])
        Diabetic=st.selectbox("your Diabetic status",['Yes','No'])
        SleepTime=st.slider("Enter your sleeptime",1.0,24.0,1.0)
        SkinCancer=st.selectbox("your SkinCancer status",['Yes','No'])
        
    with col3:
        AlcoholDrinking=st.selectbox("your alcoholdrinking status",['Yes','No'])
        MentalHealth=st.slider("Enter MentalHealth",0.0,30.0,0.1)
        AgeCategory=st.selectbox("Enter Your Age",['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'])
        PhysicalActivity=st.selectbox("your PhysicalActivity status",['Yes','No'])
        Asthma=st.selectbox("your Asthma status",['Yes','No'])
    
    
    AlcoholDrinking = 1 if(AlcoholDrinking=='Yes') else 0
    Stroke = 1 if(Stroke=='Yes') else 0
    DiffWalking = 1 if(DiffWalking=='Yes') else 0
    Sex = 1 if(Sex=='Male') else 0
    
    if(AgeCategory=='18-24'):
        AgeCategory = 0
    elif(AgeCategory=='25-29'):
        AgeCategory = 1
    elif(AgeCategory=='30-34'):
        AgeCategory = 2
    elif(AgeCategory=='35-39'):
        AgeCategory = 3
    elif(AgeCategory=='40-44'):
        AgeCategory = 4
    elif(AgeCategory=='45-49'):
        AgeCategory = 5
    elif(AgeCategory=='50-54'):
        AgeCategory = 6
    elif(AgeCategory=='55-59'):
        AgeCategory = 7
    elif(AgeCategory=='60-64'):
        AgeCategory = 8
    elif(AgeCategory=='65-69'):
        AgeCategory = 9
    elif(AgeCategory=='70-74'):
        AgeCategory = 10
    elif(AgeCategory=='75-79'):
        AgeCategory = 11
    elif(AgeCategory=='80 or older'):
        AgeCategory = 12
        
    if(Race=='White'):
        Race = 5
    elif(Race=='Black'):
        Race = 4
    elif(Race=='Asian'):
        Race = 3
    elif(Race=='American Indian/Alaskan Native'):
        Race = 2
    elif(Race=='Other'):
        Race = 1
    elif(Race=='Hispanic'):
        Race = 0
    
    Smoking = 1 if(Smoking=='Yes') else 0
    Diabetic = 1 if(Diabetic=='Yes') else 0
    PhysicalActivity = 1 if(PhysicalActivity=='Yes') else 0
    
    if(GenHealth=='Very good'):
        GenHealth = 4
    elif(GenHealth=='Fair'):
        GenHealth = 3
    elif(GenHealth=='Good'):
        GenHealth = 2
    elif(GenHealth=='Poor'):
        GenHealth = 1
    elif(GenHealth=='Excellent'):
        GenHealth = 0
    
    Asthma = 1 if(Asthma=='Yes') else 0
    KidneyDisease = 1 if(KidneyDisease=='Yes') else 0
    SkinCancer = 1 if(SkinCancer=='Yes') else 0

    #code for prediction
    if st.button("Predict"):
        result=heart_model.predict([[BMI, Smoking, AlcoholDrinking, Stroke,
       PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory,
       Race, Diabetic, PhysicalActivity, GenHealth, SleepTime,
       Asthma, KidneyDisease, SkinCancer]])
    
        if (result[0]==0):
            st.success("The preson don't have heart disease")
        else:
            st.warning("The preson have heart disease")

# Breast Cancer prediction Page
if(selected == 'Breast Cancer Prediction'):
    # Page Title
    st.title('Breast Cancer Prediction using ML')
    
    #html_temp = '''<div style ="background-color:Purple;padding:13px">
    #<h1 style ="color:white;text-align:center;"> Breast Cancer Recurrence Prediction using ML </h1>
    #</div>'''
    
    #st.markdown(html_temp, unsafe_allow_html = True)
    
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age=st.number_input("Age",min_value=0,max_value=100,step=1,placeholder='Enter your age')
        grade=st.selectbox("Tumor grade",[1,2,3])
        er=st.number_input("Enter estrogen receptors (fmol/l)",0.0,5000.0,0.1)
        
    with col2:
        meno=st.selectbox("Menopausal status",['Premenopausal','Postmenopausal'])
        nodes=st.slider("Number of positive lymph nodes",1,100,1)
        hormon=st.selectbox("Did Hormonal Therapy",['Yes','No'])
        
    with col3:
        size=st.number_input("Tumor size (in mm)",0.0,100.0,0.1)
        pgr=st.number_input("Enter progesterone receptors (fmol/l)",0.0,5000.0,0.1)
        rfstime=st.number_input("recurrence free survival time: days to first of recurrence, death or last follow-up",0,10000,1)
    
    
    meno = 1 if(meno=='postmenopausal') else 0
    hormon = 1 if(hormon=='Yes') else 0
    
    
    # creating a button for prediction
    if st.button('Predict'):
        brst_cancer_pred = brst_cancer_model.predict([[age,meno,size,grade,nodes,pgr,er,hormon,
        rfstime]])
        
        if (brst_cancer_pred[0]==0):
            st.success("The preson is alive without recurrence")
        else:
            st.warning("Breast Cancer Recurrence or Death")
        
        
# Lung Cancer prediction Page
if(selected == 'Lung Cancer Prediction'):
    
    # Upload dataset of trained model
    data = pd.read_csv("Multiple_disease_prediction/survey lung cancer.csv")

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
    
    # Page Title
    st.title('Lung Cancer Prediction using ML')
    
    #html_temp = '''<div style ="background-color:Purple;padding:13px">
    #<h1 style ="color:white;text-align:center;"> Lung Cancer Prediction using ML </h1>
    #</div>'''
    
    #st.markdown(html_temp, unsafe_allow_html = True)
    
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        GENDER=st.selectbox("Choose Your Gender",['Male','Female'])
        YELLOW_FINGERS=st.selectbox("Do you have Yellow-Fingers",['Yes','No'])
        CHRONIC_DISEASE=st.selectbox("Do you have Chronic Disease",['Yes','No'])
        WHEEZING=st.selectbox("Do you have Wheezing",['Yes','No'])
        SHORTNESS_OF_BREATH=st.selectbox("Do you have Shortness of Breath",['Yes','No'])
        
    with col2:
        AGE=st.number_input("Enter Your Age",0,100,1)
        ANXIETY=st.selectbox("Do you have Anxiety",['Yes','No'])
        FATIGUE=st.selectbox("Do you have Fatigue",['Yes','No'])
        ALCOHOL_CONSUMING=st.selectbox("Do you consume Alcohol",['Yes','No'])
        SWALLOWING_DIFFICULTY=st.selectbox("Do you have Swallowing Difficulty",['Yes','No'])
        
    with col3:
        SMOKING=st.selectbox("Do you Smoke",['Yes','No'])
        PEER_PRESSURE=st.selectbox("Do you have Peer-Pressure",['Yes','No'])
        ALLERGY=st.selectbox("Do you have Allergy",['Yes','No'])
        COUGHING=st.selectbox("Do you have Coughing Problem",['Yes','No'])
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
         
    
    # creating a button for prediction
    if st.button('Predict'):
        
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray([GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE ,
                                        ALLERGY ,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,
                                        SWALLOWING_DIFFICULTY,CHEST_PAIN])

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        
        # standardize the input data
        std_data = scaler.transform(input_data_reshaped)

        lung_cancer_diagnosis = lung_cancer_model.predict(std_data)
          
        if (lung_cancer_diagnosis[0]==0):
            st.success("No possibility of Lung cancer")
        else:
            st.warning("High possibility of Lung cancer")
    
# Parkinsons prediction Page
if(selected == 'Parkinsons Disease Prediction'):
    
    # Upload file
    data = pd.read_csv("Multiple_disease_prediction/parkinsons.csv")
    data = data.drop('name', axis=1)
    X = data.drop(columns='status',axis=1)
    Y = data['status']

    # standardize the data
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Page Title
    st.title('Parkinsons Disease Prediction using ML')
    
    #html_temp = '''<div style ="background-color:Purple;padding:13px">
    #<h1 style ="color:white;text-align:center;"> Parkinson's Disease Prediction using ML </h1>
    #</div>'''
    
    #st.markdown(html_temp, unsafe_allow_html = True)

    st.text("Enter Vocal Fundamental Frequencies below")
    
    #columns for input fields
    colA, colB, colC = st.columns(3)
    
    with colA:
        MDVP_Fo_Hz=st.number_input("Average vocal fundamental frequency MDVP-Fo(Hz)",0.0,1000.0,step=0.001,format="%.3f")
        
    with colB:
        MDVP_Fhi_Hz=st.number_input("Maximum vocal fundamental frequency MDVP-Fhi(Hz)",0.0,1000.0,step=0.001,format="%.3f")
        
    with colC:
        MDVP_Flo_Hz=st.number_input("Minimum vocal fundamental frequency MDVP-Flo(Hz)",0.0,1000.0,step=0.001,format="%.3f")
     
    st.text("Several measures of variation in fundamental frequency")
    
    colD, colE, colF = st.columns(3)
    
    with colD:
        MDVP_Jitter_persentage=st.number_input("Enter MDVP-Jitter(persentage)",placeholder="Enter value",min_value=0.0,max_value=1.0,step=0.00001,format="%.6f")
        MDVP_PPQ=st.number_input(" Enter MDVP-PPQ",0.0,10.0,step=0.00001,format="%.6f")
        
    with colE:
        MDVP_Jitter_Abs=st.number_input("Enter MDVP-Jitter(Abs) value",0.0,10.0,step=0.00001,format="%.6f")
        Jitter_DDP=st.number_input("Enter Jitter-DDP",0.0,10.0,step=0.00001,format="%.6f")
        
    with colF:
        MDVP_RAP=st.number_input("Enter MDVP-RAP",0.0,10.0,step=0.00001,format="%.6f")  
    
    st.text("Several measures of variation in amplitude")
    
    colG, colH, colI = st.columns(3)
    
    with colG:
        MDVP_Shimmer=st.number_input("Enter MDVP-Shimmer",0.0,10.0,step=0.00001,format="%.6f")
        Shimmer_APQ5=st.number_input("Enter Shimmer-APQ5",0.0,10.0,step=0.00001,format="%.6f")
        
    with colH:
        MDVP_Shimmer_dB=st.number_input("Enter MDVP-Shimmer(dB)",0.0,10.0,step=0.00001,format="%.6f")
        MDVP_APQ=st.number_input("Enter MDVP-APQ",0.0,10.0,step=0.00001,format="%.6f")
        
    with colI:
        Shimmer_APQ3=st.number_input("Enter Shimmer-APQ3",0.0,10.0,step=0.00001,format="%.6f")
        Shimmer_DDA=st.number_input("Enter Shimmer-DDA",0.0,10.0,step=0.00001,format="%.6f")
    
    st.text("Two measures of ratio of noise to tonal components in the voice")
    
    colJ, colK = st.columns(2)
    
    with colJ:
        NHR=st.number_input("Enter NHR",0.0,1000.0,step=0.00001,format="%.6f")
        
    with colK:
        HNR=st.number_input("Enter HNR",0.0,1000.0,step=0.00001,format="%.6f")
    
    st.text("Two nonlinear dynamical complexity measures")
    
    colL, colM = st.columns(2)
    
    with colL:
        RPDE=st.number_input("Enter RPDE",0.0,100.0,step=0.00001,format="%.6f")
        
    with colM:
        D2=st.number_input("Enter D2",0.0,100.0,step=0.00001,format="%.6f")
 
    st.text("Signal fractal scaling exponent")
    
    colN, colZ = st.columns(2)
    
    with colN:
        DFA=st.number_input("Enter DFA",0.0,100.0,step=0.00001,format="%.6f")
  
    st.text("Three nonlinear measures of fundamental frequency variation")
    
    colO, colP, colQ = st.columns(3)
    
    with colO:
        spread1=st.number_input("Enter spread1",-100.0,100.0,step=0.00001,format="%.6f")
        
    with colP:
        spread2=st.number_input("Enter spread2",-100.0,100.0,step=0.00001,format="%.6f")
        
    with colQ:
        PPE=st.number_input("Enter PPE",-100.0,100.0,step=0.00001,format="%.6f")

        
    if st.button("Predict"):
        # changing input data to a numpy array
        input_data_as_numpy_array = np.asarray([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_persentage,
        MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
        MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5,
        MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
    
        # reshape the numpy array
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
        # standardize the input data
        std_data = scaler.transform(input_data_reshaped)
        
        parkinsons_diagnosis = parkinsons_model.predict(std_data)
          
        if (parkinsons_diagnosis[0]==0):
            st.success("The Person does not have Parkinsons Disease")
        else:
            st.warning("The Person has Parkinsons")
        
