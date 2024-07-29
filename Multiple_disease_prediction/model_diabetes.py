# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:57:39 2024

@author: USER
"""

import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Upload file
data = pd.read_csv("diabetes.csv")
data = data.drop(columns='Outcome', axis=1)


#loading the saved model
loaded_model = pickle.load(open('C:/Users/USER/Multiple_disease_prediction/trained_model_diabetes.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing data 
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)


prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')