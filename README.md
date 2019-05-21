# Deep learning miniprojects

This repository contains our implementation of the two mendatory projects of François Chollet's _Deep Learning_ course at EPFL (EE-559).

Authors: Pierre-Antoine Desplaces and Raphaël Steinmann.

## Miniproject 1

### Project description
The goal of this project is to implement a neural network to predict the laterality of an upcoming finger movement (left or right hand) from the EEG recording 130 ms before key-press. This is a standard two-class classification problem.

The dataset comes from the _BCI competition II_ organized in May 2003 ([Benjamin Blankertz and Müller, 2002](http://www.bbci.de/competition/ii/)). It is composed of 316 training recordings, and 100 test recordings, each composed of 28 EEG channels sampled at 1khz for 0.5s.

The complete description and instructions of the first miniproject can be found under `miniproject_1 > dlc-miniproject-1.pdf`.

###


### What we did
- Normalized the data
- Linear Regression
- Logistic Regression
- Basic MLP
- Basic Convonlutional Neural Network with a simple architecture (2 conv layers w max pooling, 2 fully connected layers, tanh activations, CrossEntropyLoss)
- Better CNN with one dimensional filter

### TODO
- Do a grid search on the parameters (step size, n_epochs)
- Try different architectures, loss functions, activation functions
- Try LSTM networks
- Check https://www.quora.com/What-is-your-thought-process-when-choosing-a-neural-network-architecture
- Check which architectures are more appropriate for time series

## Miniproject 2
