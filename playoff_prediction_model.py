
#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
import keras

#%%
#string file locations
training_data_file_str = r"C:\Users\Owner\OneDrive\Desktop\CS Projects\NBA_playoff_prediction_neural_network_model\data\training\nba_season_training_data.csv"
testing_data_file_str = r"C:\Users\Owner\OneDrive\Desktop\CS Projects\NBA_playoff_prediction_neural_network_model\data\testing\nba_season_testing_data.csv"

#loading data from the csv files from their respective locations
nbapf_training_data = pd.read_csv(training_data_file_str)
nbapf_testing_data = pd.read_csv(testing_data_file_str)

vocabulary_list = nbapf_testing_data['Team'].unique()
vocabulary_dict = {k: v for v, k in enumerate(vocabulary_list)}

#separating the labels
training_data_outcome = nbapf_training_data.pop('Outcome')
testing_data_outcome = nbapf_testing_data.pop('Outcome')

nbapf_testing_data = np.array(nbapf_testing_data)
nbapf_training_data = np.array(nbapf_training_data)

for e in range(len(nbapf_training_data)):
    print(nbapf_training_data[e])
    for v in range(len(nbapf_training_data[e])):
        if nbapf_training_data[e][v] in vocabulary_dict:
           nbapf_training_data[e][v] = vocabulary_dict[nbapf_training_data[e][v]]
            
print(nbapf_training_data[0][1])

#%%

