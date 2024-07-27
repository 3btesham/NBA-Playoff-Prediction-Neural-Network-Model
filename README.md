# Predicting the Outcome of a Team in the NBA Playoffs Using a Neural Network Model
## Accuracy 7/21/2024: ~6%-50%
### Problems:
- Would choose one class to apply to every single team (ex: every team would have a first round exit or a championship)
- Probability distributions for every single team were very similar
- Sometimes would get wild outputs that made no sense (a really bad team would be declared the champion)

## Accuracy 7/25/2024: ~51%-56%
### Changes:
- Fixed multiple problems with pre-processing the data (did not apply the same process between testing/training data)
- Applied a tanh activation function rather than a relu activation funciton
- Slowed the learning rate of the adam function to 0.0005
- Adjusted the neuron architecture to have one hidden layer with 700 neurons
- Set epochs to 25

### Problems:
- Appears to only choose first round exit for worse teams, second round exit for better teams, conference finals very very rarely, championship for teams that are far better than other teams
- Doesn't choose any team for finals exit
- Overfitting majority of the time

### Possible Improvements:
- Specify the conference of each team and standing within conference (better reflect reality of the playoffs)
- Add net difference for each stat from the league average
- Play around with different optimization functions and different loss functions
- Adjust hyperparameters

## Accuracy 7/27/2024: ~52%-61%
### Changes:
- Added team conference and team standing in conference to training and testing data
- Used .map() to map categorical data in teams and conference
- Updated pre-processing of data to accomodate for standings and conferences
- Made neural network 5 layers (1 input, 3 hidden, 1 output) and had decreasing neurons in each hidden layer (1/3 of total input shape decreased)
- Added a tanh activation function to the first hidden layer, no activation functions for other layers

### Problems:
- When examining all training data output, the model apears to not choose a championship unitl reaching later season (usually 2015 up)
- Prior to 2015 the model will usually choose teams to have a conference finals exit, afterwards the model will not choose many conference finals teams exits
- The model never chooses a team for a finals exit
- The model will choose more than one championship winner in a single season
- Still far below the goal of a 90% accuracy model

### Possible Improvements:
- Add total playoff games, playoff wins, losses for all players for each team
- Add total championships for all players for each team
- Add league average net differentials for each datapoint for every single statistic for every single team
- Research other advanced statistics that accurately evaluate a teams possible performance in the playoffs
- Research how to limit choices for outcome within a certain batch?
- Figure out how to add player data for every single team???
