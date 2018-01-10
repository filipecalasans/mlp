# MLP: Multilayer Perceptron

Multilayer Perceptron implementation using Python. This MLP implementation uses quadratic cost function and sigmoid activation function. Users are able to customized the activation code, however we still don't give support for changing the cost function.

# Examples

## XOR Gate

In this example, we trained a Neural Network (NN) to learn the logic XOR gate. We train the NN using the stochastic gradient descent. The example is located in the file *xor-mlp.py* and the dataset in the file *XOR.dat*

## Iris UCI Data Set

In this example, we trained a NN using the dataset Iris that is provided at [UCI: Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html). This specific example validates the mini-batch gradient descent algoithm.

We tunned the NN training parameters in order to get successful. Therefore, if you want to save time reproducing this example you should use *eta < 0.1*. After tunning, we found out that *eta=0.05* and mini-batch size equal to 10 converges at a reasonable speed.

The source code is provided *iris-mlp-minibatch.py* and dataset in the file *iris.data*.

### Cross-Entropy Cost Function

The source file *cross-entropy-cost-function.py* implements Cross-Entropy Cost function and applyies on the Iris UCI Data Set. We show in this example that in order to implement a different cost functions, users must reimplement two methods from *NeuralNetwork* class: 
   
```python
   #
   # a: n x 1 column numpy matrix (Neural Network output in the output layer - estimated values)
   # y: n x 1 column numpy matrix (Test Example desired output)
   #    
   def cost_function(self, a, y)

   def cost_function_gradient(self, a, y)
```


