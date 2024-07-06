#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

#%%
#string file locations
training_data_file_str = r"C:\Users\Owner\OneDrive\Desktop\CS Projects\NBA_playoff_prediction_neural_network_model\data\training\nba_season_training_data.csv"
testing_data_file_str = r"C:\Users\Owner\OneDrive\Desktop\CS Projects\NBA_playoff_prediction_neural_network_model\data\testing\nba_season_testing_data.csv"

#loading data from the csv files from their respective locations
nbapf_training_data = pd.read_csv(training_data_file_str)
nbapf_testing_data = pd.read_csv(testing_data_file_str)

vocabulary_list = list(nbapf_training_data['Team'].unique())
for i in nbapf_testing_data['Team']:
    if i not in vocabulary_list:
        vocabulary_list.append(i)

vocabulary_dict = {k: v for v, k in enumerate(vocabulary_list)}

#separating the labels
training_data_outcome = nbapf_training_data.pop('Outcome')
testing_data_outcome = nbapf_testing_data.pop('Outcome')

nbapf_testing_data = np.array(nbapf_testing_data)
nbapf_training_data = np.array(nbapf_training_data)
print(nbapf_training_data.shape)

class_names = ['First Round Exit', 'Second Round Exit', 'Conference Finals Exit', 'Finals Exit', 'Championship']

#%%
#preprocessing all the data
for e in range(len(nbapf_training_data)):
    if nbapf_training_data[e][1] in vocabulary_dict:
        nbapf_training_data[e][1] = vocabulary_dict[nbapf_training_data[e][1]] #team
    
    nbapf_training_data[e][2] /= 82.0 #wins
    nbapf_training_data[e][3] /= 16.0 #win ranks
    nbapf_training_data[e][4] /= 12.28 #MOV
    nbapf_training_data[e][5] /= 16.0 #MOV_Rank
    nbapf_training_data[e][6] /= 11.92 #SRS
    nbapf_training_data[e][7] /= 16.0 #SRS_Rank
    nbapf_training_data[e][8] /= 123.2 #ORTG
    nbapf_training_data[e][9] /= 16.0 #ORTG_Rank
    nbapf_training_data[e][10] /= 91.3 #DRTG
    nbapf_training_data[e][11] /= 16.0 #DRTG_Rank
    nbapf_training_data[e][12] /= 13.4 #NRTG
    nbapf_training_data[e][13] /= 16.0 #NRTG_Rank
    nbapf_training_data[e][14] /= 57.8 #eFG
    nbapf_training_data[e][15] /= 16.0 #eFG_Rank
    nbapf_training_data[e][16] /= 42.0 #Opp_eFG
    nbapf_training_data[e][17] /= 16.0 #Opp_eFG_Rank

for e in range(len(nbapf_testing_data)):
    if nbapf_testing_data[e][1] in vocabulary_dict:
        nbapf_testing_data[e][1] = vocabulary_dict[nbapf_testing_data[e][1]] #team
    
    nbapf_testing_data[e][2] /= 82.0 #wins
    nbapf_testing_data[e][3] /= 16.0 #win ranks
    nbapf_testing_data[e][4] /= 12.28 #MOV
    nbapf_testing_data[e][5] /= 16.0 #MOV_Rank
    nbapf_testing_data[e][6] /= 11.92 #SRS
    nbapf_testing_data[e][7] /= 16.0 #SRS_Rank
    nbapf_testing_data[e][8] /= 123.2 #ORTG
    nbapf_testing_data[e][9] /= 16.0 #ORTG_Rank
    nbapf_testing_data[e][10] /= 91.3 #DRTG
    nbapf_testing_data[e][11] /= 16.0 #DRTG_Rank
    nbapf_testing_data[e][12] /= 13.4 #NRTG
    nbapf_testing_data[e][13] /= 16.0 #NRTG_Rank
    nbapf_testing_data[e][14] /= 57.8 #eFG
    nbapf_testing_data[e][15] /= 16.0 #eFG_Rank
    nbapf_testing_data[e][16] /= 42.0 #Opp_eFG
    nbapf_testing_data[e][17] /= 16.0 #Opp_eFG_Rank

nbapf_training_data = tf.convert_to_tensor(nbapf_training_data, tf.float32)
nbapf_testing_data = tf.convert_to_tensor(nbapf_testing_data, tf.float32)


#%%
#creating the model
playoff_predictor = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

#%%
#Compiling the model
playoff_predictor.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#%%
playoff_predictor.fit(
    nbapf_training_data, 
    training_data_outcome,
    shuffle = False,
    batch_size=16, 
    epochs=10
    )

#%%
#evaluating the model
test_loss, test_acc = playoff_predictor.evaluate(nbapf_testing_data, testing_data_outcome, verbose=1, batch_size=16)
print('Test Accuracy: ', test_acc)

# %%
#predicting the model
predictions = playoff_predictor.predict(nbapf_testing_data)
for i in range(len(predictions)):
    print(int(nbapf_testing_data[i][0]), " ", vocabulary_list[int(nbapf_testing_data[i][1])], "is expected to have a", class_names[np.argmax(predictions[i])-1])
    print('They actually had a ' + class_names[testing_data_outcome[i]-1])
    
# %%
