import torch
from torch import Tensor
from torch.autograd import Variable

import numpy as np

torch.manual_seed(1)
np.random.seed(1)
torch.set_num_threads(4)

import dlc_bci as bci
from helpers import *
from modules import *


print("Importing and preprocessing the data...")
# import full data sampled at 1Khz
train_input , train_target = bci.load(root = './data_bci', one_khz = True)
test_input , test_target = bci.load(root = './data_bci', train = False, one_khz = True)

# normalization
train_mean = train_input.mean(2).mean(0).unsqueeze(1).expand(-1,train_input.size(2))
train_std = train_input.std(2).std(0).unsqueeze(1).expand(-1,train_input.size(2))

test_mean = test_input.mean(2).mean(0).unsqueeze(1).expand(-1,test_input.size(2))
test_std = test_input.std(2).std(0).unsqueeze(1).expand(-1,test_input.size(2))

train_input.sub_(train_mean)
test_input.sub_(test_mean)

train_input.div_(train_input.std())
test_input.div_(test_input.std())

# data augmentation by slicing
seq = [ train_input[:,:,i::10] for i in range(10) ]
train_input = torch.cat(seq)

train_target = torch.cat([train_target]*10)

seq = [ test_input[:,:,i::10] for i in range(10) ]
test_input = torch.cat(seq)

test_target = torch.cat([test_target]*10)

# split train/validation set
split = 2500
indices = np.arange(train_input.size(0))
np.random.shuffle(indices)

full_input = train_input.clone()
train_input = full_input[indices[:split].tolist()]
validation_input = full_input[indices[split:].tolist()]

full_target = train_target.clone()
train_target = full_target[indices[:split].tolist()]
validation_target = full_target[indices[split:].tolist()]


print("Train input :", str(type(train_input)), train_input.size()) 
print("Train target :", str(type(train_target)), train_target.size())
print("Validation input :", str(type(validation_input)), validation_input.size()) 
print("Validation target :", str(type(validation_target)), validation_target.size())
print("Test input :", str(type(test_input)), test_input.size()) 
print("Test target :", str(type(test_target)), test_target.size())


best_model = CNN_1D()
best_epoch = 20
best_lr = 0.005
mini_batch_size = 250

print("Training the model for", best_epoch, "epochs, with a learning rate of", best_lr)
train_model(best_model, Variable(train_input), Variable(train_target), mini_batch_size, nb_epochs=best_epoch, learning_rate=best_lr)
nb_error = compute_nb_errors(best_model, Variable(test_input), Variable(test_target))
test_error = 100*(nb_error/test_input.size(0))
print("Test error =", test_error, "%")


