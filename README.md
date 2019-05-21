# Deep learning miniprojects

This repository contains our implementation of the two mendatory projects of François Chollet's _Deep Learning_ course at EPFL (EE-559).

Authors: Pierre-Antoine Desplaces and Raphaël Steinmann.

## Miniproject 1: Finger movement prediction from encephalogram data

### Project description
The goal of this project is to implement a neural network to predict the laterality of an upcoming finger movement (left or right hand) from the EEG recording 130 ms before key-press. This is a standard two-class classification problem.

The dataset comes from the _BCI competition II_ organized in May 2003 ([Benjamin Blankertz and Müller, 2002](http://www.bbci.de/competition/ii/)). It is composed of 316 training recordings, and 100 test recordings, each composed of 28 EEG channels sampled at 1khz for 0.5s.

The complete description and instructions of the first miniproject can be found under `miniproject_1 > project1_instructions.pdf`.

### Experiments and results

The experiments and the results are described in `miniproject_1 > project1_report.pdf` and the best result can be reproduced by running `test.py` with no arguments.

## Miniproject 2: Implementation of a basic deep learning framework from scratch 

### Project description
The objective of this project is to design a mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.

The following modules were implemented:
- Tanh an ReLU activations,
- Fully connected layers,
- MSE and cross-entropy losses,
- SGD optimizer,
- A sequential module to combine several modules in a basic sequential structure.

The complete description and instructions of the first miniproject can be found under `miniproject_2 > project2_instructions.pdf`.

### Experiments and results

The experiments and the results are described in `miniproject_2 > project2_report.pdf` and the best result can be reproduced by running `test.py` with no arguments.
