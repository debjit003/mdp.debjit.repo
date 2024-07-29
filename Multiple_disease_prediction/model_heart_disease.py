# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:21:12 2024

@author: USER
"""

import numpy as np
import pickle


#loading the saved model
loaded_model = pickle.load(open('C:/Users/USER/Multiple_disease_prediction/trained_model_heartDisease.sav', 'rb'))

input_data = (57,1,0,140,192,0,1,148,0,0.4,1,0,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person does not have heart disease')
else:
  print('The person have heart disease')