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

### Possible improvements:
- Specify the conference of each team and standing within conference (better reflect reality of the playoffs)
- Add net difference for each stat from the league average
- Play around with different optimization functions and different loss functions
- Adjust hyperparameters

## Accuracy 7/27/2024: 
