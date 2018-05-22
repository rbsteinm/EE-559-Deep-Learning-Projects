import torch
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
            print("    Epoch", e+1, "loss =", sum_loss)

def compute_nb_errors(model, data_input, data_target):
    nb_data_errors = 0
    model.eval()
    output = model(data_input)
    _, predicted_classes = torch.max(output.data, 1)
    for k in range(data_target.size(0)):
        if data_target.data[k] != predicted_classes[k]:
            nb_data_errors = nb_data_errors + 1
    return nb_data_errors

# grid search for parameters on validation set
def find_best_params(network, train_input, train_target, val_input, val_target, learning_rates, nb_epochs_total, epoch_step, mini_batch_size):
    res = []
    best_error = val_input.size(0)+1

    print("MODEL", network().__class__.__name__)
    print("="*18)

    for lr in learning_rates: 
        print("  Learning rate", lr, ":")
        print("-"*23)

        epoch_acc = []
        model = network()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for e in range(nb_epochs_total):
            sum_loss = 0
            model.train()
            for b in range(0, train_input.size(0), mini_batch_size):
                output = model(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                sum_loss = sum_loss + loss.data[0]
                model.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (e+1)%epoch_step == 0 or e==0:
                validation_error = compute_nb_errors(model, val_input, val_target)
                epoch_acc.append(100*(validation_error/val_input.size(0)))

                if (e+1)%(10*epoch_step) == 0 or e==0:
                    print("    Epoch {:>4} : loss = {:1.8f} | validation error = {:>4}".format(e+1, sum_loss, validation_error))
                
                if (validation_error < best_error) or ( (validation_error == best_error) and ((e+1) < best_epoch) ):
                    best_error = validation_error
                    best_epoch = e+1
                    best_lr = lr
                    
        res.append(epoch_acc)
    print("Done.")
    return res, best_epoch, best_lr, best_error
