import torch
import math

from torch import optim
from torch import Tensor
from torch import nn


# takes continuous predicitions
# returns discrete values in {0,1}
def discrete_predictions(preds):
        return preds.apply_(lambda x: 1 if x >= 0.5 else 0).long()

# takes two tensors as input with values in {0, 1}
# returns the accuracy of the predictions between 0 and 1
def compute_accuracy(labels, predictions):
    n_errors = (predictions - labels).abs().sum()
    error_rate = n_errors / labels.size(0)
    return 1-error_rate


def train_model(model, train_input, train_target, mini_batch_size = 100, nb_epochs = 250):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    for e in range(0, nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            print(train_target.narrow(0, b, mini_batch_size))
            print(output)
            #print(train_target.narrow(0, b, mini_batch_size)[0])
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size).unsqueeze(1))
            model.zero_grad()
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, data_input, data_target):
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(0, mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors