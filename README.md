# MLP: Multilayer Perceptron

Multilayer perceptron implementation using Python and Numpy. We use the XOR gate example to test
the network.

# Examples Provided

We provide the library with two examples. The XOR gate and the UCI iris dataset.

## XOR Gate

In this example, we traing a Neural Network (NN) to learn the logic XOR gate. We train the NN using the stochastic gradient descent. The example is located in the file *xor-mlp.py*

The dataset is provided in the file *XOR.dat*

## Iris UCI Data Set

In this example, we train a NN using the dataset Iris provided at [UCI: Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html). This specific example validate the mini-batch gradient descent.

We did some tunning in the nn training parameters. Therefore, if you want to reproduce this example you must use *eta < 0.1*. After tunning, we found out that *eta=0.05* and mini-batch size equal to 10 converges at a reasonable speed.

The dataset is provided in the file *iris.data*.