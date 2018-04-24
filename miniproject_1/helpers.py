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


def train_model(model, train_input, train_target, nb_epochs = 10):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = 1e-3)
	for e in range(0, nb_epochs):
		output = model(train_input)
		print(output.size())
		print(train_target.size())
		loss = criterion(output, train_target)
		model.zero_grad()
		loss.backward()
		optimizer.step()

def compute_nb_errors(model, data_input, data_target):
	nb_data_errors = 0

	output = model(data_input)
	_, predicted_classes = torch.max(output.data, 1)
	for k in range(0, data_target.size(0)):
		if data_target.data[k] != predicted_classes[k]:
			nb_data_errors = nb_data_errors + 1

	return nb_data_errors