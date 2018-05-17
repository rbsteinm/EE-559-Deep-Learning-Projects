from torch import FloatTensor, LongTensor, Tensor
import math

def generate_disc_set(nb):
    a = Tensor(nb, 2).uniform_(0, 1)
    target = ((a.pow(2).sum(1)).sqrt() < math.sqrt(1/(2*math.pi))).long()
    return a, target


# converts 'target' Tensor to one hot labels
def convert_to_one_hot(target):
    tmp = FloatTensor(target.size(0), 2).fill_(0)
    for k in range(0, target.size(0)):
        tmp[k, target[k]] = 1
    return tmp


# runs the input through the model and computes
# the accuracy of its predictions against the labels
def compute_accuracy(model, input_, target):
    nb_data_errors = 0
    output = model.forward(input_)

    _, predicted_classes = output.max(1)
    for k in range(input_.size(0)):
        if target[k] != predicted_classes[k]:
            nb_data_errors = nb_data_errors + 1
    return 100 - (100*(nb_data_errors / input_.size(0)))


# trains the model with the training data given in input
def train_model(train_input, train_one_hot_target, model, criterion, optimizer, nb_epochs=100, mini_batch_size=5, verbose=False):
    for e in range(0, nb_epochs):
        loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            # sum the loss for each batch to get the current epoch's loss
            loss += criterion.forward(output, train_one_hot_target.narrow(0, b, mini_batch_size))
            # set the gradients of all layers to zero before the next batch can go through the network
            model.zero_grad()
            model.backward(criterion.backward())
            optimizer.step() # performs a gradient step to optimize the parameters
        if verbose:
            print("Epoch", e+1, ":", loss)