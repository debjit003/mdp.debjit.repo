# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:20:46 2024

@author: USER
"""
import streamlit as st
from warnings import simplefilter
import pickle

simplefilter(action='ignore', category=FutureWarning)


#loading the saved model
loaded_model = pickle.load(open('C:/Users/USER/Multiple_disease_prediction/trained_model_heartDisease.sav', 'rb'))


#data=pd.read_csv('C:/Users/USER/Multiple_disease_prediction/heart_2020_cleaned.csv')
#data['Smoking']=data['Smoking'].map({'Yes':1,'No':0})
#data['AlcoholDrinking']=data['AlcoholDrinking'].map({'Yes':1,'No':0})
#data['Stroke']=data['Stroke'].map({'Yes':1,'No':0})
#data['DiffWalking']=data['DiffWalking'].map({'Yes':1,'No':0})
#data['Diabetic']=data['Diabetic'].map({'Yes':1,'No':0})
#data['Asthma']=data['Asthma'].map({'Yes':1,'No':0})
#data['KidneyDisease']=data['KidneyDisease'].map({'Yes':1,'No':0})
#data['SkinCancer']=data['SkinCancer'].map({'Yes':1,'No':0})
#data['PhysicalActivity']=data['PhysicalActivity'].map({'Yes':1,'No':0})
#data['Sex']=data['Sex'].map({'Male':1,'Female':0})
#data['Race']=data['Race'].map({'White':5, 'Black':4, 'Asian':3, 'American Indian/Alaskan Native':2,'Other':1, 'Hispanic':0})
#data['GenHealth']=data['GenHealth'].map({'Very good':4, 'Fair':3, 'Good':2, 'Poor':1, 'Excellent':0})
#data.replace('80 or older','80-85',inplace=True)
#data.replace('80 or older','80-85',inplace=True)

#le = LabelEncoder()
#data['Sex']=le.fit_transform(data['Sex'])
#data['AgeCategory']=le.fit_transform(data['AgeCategory'])
#data['Race']=le.fit_transform(data['Race'])
#data['Diabetic']=le.fit_transform(data['Diabetic'])
#data['GenHealth']=le.fit_transform(data['GenHealth'])


#X=data.drop(['HeartDisease'],axis=1).values
#y=data['HeartDisease']

#model=loaded_model(random_state=0)
#model.fit(X,y)



#image=Image.open('heart_attack.jpeg')


def main():
    #st.title("Heart disease predictions ")

    html_temp = '''<div style ="background-color:Purple;padding:13px">
    <h1 style ="color:white;text-align:center;"> Heart Disease Prediction using ML </h1>
    </div>'''
    
    st.markdown(html_temp, unsafe_allow_html = True)



    BMI=st.slider("Enter body mass index",12.02,94.85)
    Smoking=st.selectbox("your Smoking status",['Yes','No'])
    AlcoholDrinking=st.selectbox("your alcoholdrinking status",['Yes','No'])
    Stroke=st.selectbox("your Stroke status",['Yes','No'])
    PhysicalHealth=st.slider("Enter PhysicalHealth",0.0,30.0,0.1)
    MentalHealth=st.slider("Enter MentalHealth",0.0,30.0,0.1)
    DiffWalking=st.selectbox("your DiffWalking status",['Yes','No'])
    Sex=st.selectbox("Gender",['Male','Female'])
    AgeCategory=st.selectbox("Enter Your Age",['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'])
    Race=st.selectbox("Ethnicity",['White','Black','Asian','American Indian/Alaskan Native','Other','Hispanic'])
    Diabetic=st.selectbox("your Diabetic status",['Yes','No'])
    PhysicalActivity=st.selectbox("your PhysicalActivity status",['Yes','No'])
    GenHealth=st.selectbox("your GenHealth status",['Excellent','Very good','Fair','Good','Poor'])
    SleepTime=st.slider("Enter your sleeptime",1.0,24.0,1.0)
    Asthma=st.selectbox("your Asthma status",['Yes','No'])
    KidneyDisease=st.selectbox("your KidneyDiseas status",['Yes','No'])
    SkinCancer=st.selectbox("your SkinCancer status",['Yes','No'])
    
    
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

    
    if st.button("Predict"):
        result=loaded_model.predict([[BMI, Smoking, AlcoholDrinking, Stroke,
       PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory,
       Race, Diabetic, PhysicalActivity, GenHealth, SleepTime,
       Asthma, KidneyDisease, SkinCancer]])
    
        if (result[0]==0):
            st.success("The preson don't have heart disease")
        else:
            st.warning("The preson have heart disease")
    



if __name__=='__main__':
    main()

