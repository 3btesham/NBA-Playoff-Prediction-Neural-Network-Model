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

training_season = nbapf_training_data.pop('Season')
testing_season = nbapf_testing_data.pop('Season')

#separating the labels
training_data_outcome = nbapf_training_data.pop('outcome')
testing_data_outcome = nbapf_testing_data.pop('outcome')

nbapf_training_data['Team'] = nbapf_training_data['Team'].map(vocabulary_dict)
nbapf_training_data['Conference'] = nbapf_training_data['Conference'].map({'West': 1, 'East': 0})
nbapf_testing_data['Team'] = nbapf_testing_data['Team'].map(vocabulary_dict)
nbapf_testing_data['Conference'] = nbapf_testing_data['Conference'].map({'West': 1, 'East': 0})
print(nbapf_training_data.tail())

nbapf_testing_data = np.array(nbapf_testing_data)
training_data_outcome = np.array(training_data_outcome)
nbapf_training_data = np.array(nbapf_training_data)
testing_data_outcome = np.array(testing_data_outcome)

class_names = ['First Round Exit', 'Second Round Exit', 'Conference Finals Exit', 'Finals Exit', 'Championship']

#%%
#preprocessing all the data

for e in range(len(nbapf_training_data)):    
    nbapf_training_data[e][2] /= 32.0 #age
    nbapf_training_data[e][3] /= 82.0 #wins
    nbapf_training_data[e][4] /= 8.0 #standing
    nbapf_training_data[e][5] /= 12.28 #MOV
    nbapf_training_data[e][7] /= 11.92 #SRS
    nbapf_training_data[e][8] /= 123.2 #ORTG
    nbapf_training_data[e][9] /= 130 #DRTG
    nbapf_training_data[e][10] /= 13.4 #NRTG
    nbapf_training_data[e][11] /= 113.68 #Pace
    nbapf_training_data[e][16] /= 100.0 #TOV%
    nbapf_training_data[e][17] /= 100.0 #ORB%
    nbapf_training_data[e][20] /= 100.0 #Opp_TOV%
    nbapf_training_data[e][21] /= 100.0 #DRB%
    nbapf_training_data[e][23] /= 49.9 #FG
    nbapf_training_data[e][24] /= 119.6 #FGA
    nbapf_training_data[e][26] /= 45.4 #3PM
    nbapf_training_data[e][27] /= 45.4 #3PA
    nbapf_training_data[e][29] /= 80.0 #2PM
    nbapf_training_data[e][30] /= 119.6 #2PA
    nbapf_training_data[e][32] /= 31.9 #FT
    nbapf_training_data[e][33] /= 42.4 #FTA
    nbapf_training_data[e][35] /= 18.5 #ORB
    nbapf_training_data[e][36] /= 42.2 #DRB
    nbapf_training_data[e][37] /= 80.2 #TRB
    nbapf_training_data[e][38] /= 31.4 #AST
    nbapf_training_data[e][39] /= 12.9 #STL
    nbapf_training_data[e][40] /= 8.7 #BLK
    nbapf_training_data[e][41] /= 24.5 #TOV
    nbapf_training_data[e][42] /= 32.1 #PF
    nbapf_training_data[e][43] /= 126.5 #PTS
    nbapf_training_data[e][44] /= 52.01 #OppFGM
    nbapf_training_data[e][45] /= 103.7 #OppFGA
    nbapf_training_data[e][47] /= 14.83 #Opp3PM
    nbapf_training_data[e][48] /= 40.59 #Opp3PA
    nbapf_training_data[e][50] /= 41.2 #Opp2PM
    nbapf_training_data[e][51] /= 80.1 #Opp2PA
    nbapf_training_data[e][53] /= 28.99 #OppFT
    nbapf_training_data[e][54] /= 37.51 #OppFTA
    nbapf_training_data[e][56] /= 18.6 #OppORB
    nbapf_training_data[e][57] /= 37.89 #OppDRB
    nbapf_training_data[e][58] /= 59.57 #OppTRB
    nbapf_training_data[e][59] /= 30.94 #OppAST
    nbapf_training_data[e][60] /= 11.65 #OppSTL
    nbapf_training_data[e][61] /= 7.98 #OppBLK
    nbapf_training_data[e][62] /= 24.15 #OppTOV
    nbapf_training_data[e][63] /= 29.91 #OppPF
    nbapf_training_data[e][64] /= 130.77 #OppPTS
    nbapf_training_data[e][65] /= 7.0 #allstars

for e in range(len(nbapf_testing_data)):
    if nbapf_testing_data[e][0] in vocabulary_dict:
        nbapf_testing_data[e][0] = vocabulary_dict[nbapf_testing_data[e][0]] #team
    
    nbapf_testing_data[e][2] /= 32.0 #age
    nbapf_testing_data[e][3] /= 82.0 #wins
    nbapf_testing_data[e][4] /= 8.0 #standing
    nbapf_testing_data[e][5] /= 12.28 #MOV
    nbapf_testing_data[e][7] /= 11.92 #SRS
    nbapf_testing_data[e][8] /= 123.2 #ORTG
    nbapf_testing_data[e][9] /= 130 #DRTG
    nbapf_testing_data[e][10] /= 13.4 #NRTG
    nbapf_testing_data[e][11] /= 113.68 #Pace
    nbapf_testing_data[e][16] /= 100.0 #TOV%
    nbapf_testing_data[e][17] /= 100.0 #ORB%
    nbapf_testing_data[e][20] /= 100.0 #Opp_TOV%
    nbapf_testing_data[e][21] /= 100.0 #DRB%
    nbapf_testing_data[e][23] /= 49.9 #FG
    nbapf_testing_data[e][24] /= 119.6 #FGA
    nbapf_testing_data[e][26] /= 45.4 #3PM
    nbapf_testing_data[e][27] /= 45.4 #3PA
    nbapf_testing_data[e][29] /= 80.0 #2PM
    nbapf_testing_data[e][30] /= 119.6 #2PA
    nbapf_testing_data[e][32] /= 31.9 #FT
    nbapf_testing_data[e][33] /= 42.4 #FTA
    nbapf_testing_data[e][35] /= 18.5 #ORB
    nbapf_testing_data[e][36] /= 42.2 #DRB
    nbapf_testing_data[e][37] /= 80.2 #TRB
    nbapf_testing_data[e][38] /= 31.4 #AST
    nbapf_testing_data[e][39] /= 12.9 #STL
    nbapf_testing_data[e][40] /= 8.7 #BLK
    nbapf_testing_data[e][41] /= 24.5 #TOV
    nbapf_testing_data[e][42] /= 32.1 #PF
    nbapf_testing_data[e][43] /= 126.5 #PTS
    nbapf_testing_data[e][44] /= 52.01 #OppFGM
    nbapf_testing_data[e][45] /= 103.7 #OppFGA
    nbapf_testing_data[e][47] /= 14.83 #Opp3PM
    nbapf_testing_data[e][48] /= 40.59 #Opp3PA
    nbapf_testing_data[e][50] /= 41.2 #Opp2PM
    nbapf_testing_data[e][51] /= 80.1 #Opp2PA
    nbapf_testing_data[e][53] /= 28.99 #OppFT
    nbapf_testing_data[e][54] /= 37.51 #OppFTA
    nbapf_testing_data[e][56] /= 18.6 #OppORB
    nbapf_testing_data[e][57] /= 37.89 #OppDRB
    nbapf_testing_data[e][58] /= 59.57 #OppTRB
    nbapf_testing_data[e][59] /= 30.94 #OppAST
    nbapf_testing_data[e][60] /= 11.65 #OppSTL
    nbapf_testing_data[e][61] /= 7.98 #OppBLK
    nbapf_testing_data[e][62] /= 24.15 #OppTOV
    nbapf_testing_data[e][63] /= 29.91 #OppPF
    nbapf_testing_data[e][64] /= 130.77 #OppPTS
    nbapf_testing_data[e][65] /= 7.0 #allstars

for e in range(len(training_data_outcome)):
    training_data_outcome[e] -= 1

for e in range(len(testing_data_outcome)):
    testing_data_outcome[e] -= 1

print(nbapf_training_data[0])

#%%
#Convert data to tensors
nbapf_training_data = tf.convert_to_tensor(nbapf_training_data, tf.float32)
nbapf_testing_data = tf.convert_to_tensor(nbapf_testing_data, tf.float32)

#%%
#creating the model
playoff_predictor = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1050,activation='tanh'),
    keras.layers.Dense(700),
    keras.layers.Dense(350),
    keras.layers.Dense(5, activation='softmax')
])

#%%
#Compiling the model
playoff_predictor.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#%%
playoff_predictor.fit(
    nbapf_training_data, 
    training_data_outcome,
    shuffle = True,
    batch_size=16, 
    epochs=25
    )

#%%
#evaluating the model
test_loss, test_acc = playoff_predictor.evaluate(nbapf_testing_data, testing_data_outcome, verbose=1, batch_size=16)
print('Test Accuracy: ', test_acc)

# %%
#predicting the model
predictions = playoff_predictor.predict(nbapf_testing_data, batch_size=16)
for i in range(160, 176):
    print(int(testing_season[i]), " ", vocabulary_list[int(nbapf_testing_data[i][0])], "are expected to have a", class_names[np.argmax(predictions[i])])
    print('They actually had a ' + class_names[testing_data_outcome[i]])
    
# %%
#predicting the playoffs 2
predictions = playoff_predictor.predict(nbapf_testing_data, batch_size=16)
for i in range(160, 176):
    print(int(testing_season[i]), vocabulary_list[int(nbapf_testing_data[i][0])])
    print(predictions[i])
# %%
