from helpers import *
from framework import *

# generate training and testing data
train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

# convert targets to one hot labels
train_one_hot_target = convert_to_one_hot(train_target)
test_one_hot_target = convert_to_one_hot(test_target)

# initialize model and parameters
nb_epochs = 100
mini_batch_size = 5
model = Sequential(Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25, 25), Tanh(), Linear(25,2))
criterion = LossMSE()
#criterion = LossCrossEntropy()
optimizer = optim_SGD(model.param(), 1e-3)

# train the model
train_model(train_input, train_one_hot_target, model, criterion, optimizer, nb_epochs, mini_batch_size, verbose=True)

# compute train and test accuracy
print("Train accuracy :", compute_accuracy(model, train_input, train_target), "%")
print("Test accuracy :", compute_accuracy(model, test_input, test_target), "%")