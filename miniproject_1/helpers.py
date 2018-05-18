import torch
import math

from torch import optim
from torch import Tensor, LongTensor
from torch import nn

# takes continuous predicitions
# returns discrete values in {0,1}
def discrete_predictions(preds):
        return preds.apply_(lambda x: 1 if x >= 0.5 else 0).long()

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs=100, learning_rate=1e-3, verbose=True):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for e in range(0, nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.data[0]
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and (((e+1)%10)==0 or e==0):
            print("Epoch", e+1, "loss =", sum_loss)

def compute_nb_errors(model, data_input, data_target):
    nb_data_errors = 0
    model.eval()
    output = model(data_input)
    _, predicted_classes = torch.max(output.data, 1)
    for k in range(data_target.size(0)):
        if data_target.data[k] != predicted_classes[k]:
            nb_data_errors = nb_data_errors + 1
    return nb_data_errors

def convert_to_one_hot_labels(target):
    tmp = LongTensor(target.size(0), target.max() + 1).fill_(-1)
    for k in range(target.size(0)):
        tmp[k, target[k]] = 1
    return tmp
