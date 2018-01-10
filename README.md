# MLP: Multilayer Perceptron

Multilayer Perceptron implementation using Python. This MLP implementation uses quadratic cost function and sigmoid activation function. Users are able to customized the activation code, however I still don't give support for changing the cost function.

# Examples

## XOR Gate

In this example, I trained a Neural Network (NN) to learn the logic XOR gate. I trained the NN using the stochastic gradient descent. The example is located in the file *xor-mlp.py* and the dataset in the file *XOR.dat*

## Iris UCI Data Set

In this example, I trained a NN using the dataset Iris that is provided at [UCI: Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html). This specific example validates the mini-batch gradient descent algoithm.

I tunned the NN hyperparameters in order to get successful. Therefore, if you want to save time reproducing this example you should use *eta < 0.1*. After tunning, I found out that *eta=0.05* and mini-batch size equal to 10 converges at a reasonable speed.

The source code is provided in *iris-mlp-minibatch.py* and dataset in the file *iris.data*.

### Cross-Entropy Cost Function (Switching the Neural Network Cost Function)

The source file *cross-entropy-cost-function.py* implements Cross-Entropy Cost function and applyies the NN over the Iris Data Set. I show in this example that in order to implement a different cost functions, users must reimplement two methods from *NeuralNetwork* class: 
   
```python
   
class NeuralNetwork(object):   
   #
   # a: n x 1 column numpy matrix (Neural Network output in the output layer - estimated values)
   # y: n x 1 column numpy matrix (Test Example desired output)
   #    
   def cost_function(self, a, y)

   def cost_function_gradient(self, a, y)
```

You can use two approaches: You can reassign the two methods on a *NeuralNetwork* instance, as shown below, or you can inherit *NeuralNetwork* class and override both methods.

```python
   
   nn_size = [3, 3, 1]

   mlp = NeuralNetwork(nn_size)
   
   # 
   #  Implement cross-entropy cost function: (Sum for each output neuron)
   #  C = -sum([yln(a) + (1-y)ln(1-a)])
   #  gradient(C) = (a-y)/(a(1-a)) 
   def cross_entropy_gradient(a,y):
      return (a-y)/(a*(np.ones(a.shape)-a))

   def cross_entropy(a,y):
      return -1*np.sum(y*np.log(a) + ((np.ones(y.shape)-y)*np.log(np.ones(a.shape)-a)))
   
   mlp.cost_function = cross_entropy 
   mlp.cost_function_gradient = cross_entropy_gradient
```


