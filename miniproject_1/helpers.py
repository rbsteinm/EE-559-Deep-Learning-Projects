# takes continuous predicitions
# returns discrete values in {0,1}
def discrete_predictions(preds):
        return preds.apply_(lambda x: 1 if x >= 0.5 else 0).longx§()

# takes two tensors as input with values in {0, 1}
# returns the accuracy of the predictions between 0 and 1
def compute_accuracy(labels, predictions):
    n_errors = (predictions - labels).abs().sum()
    error_rate = n_errors / labels.size(0)
    return 1-error_rate