# MLP: Multilayer Perceptron

Multilayer Perceptron implementation using Python. The default MLP implementation uses quadratic cost function and sigmoid activation function. Users are able to customize the activation code and cost function as needed.

# Examples

## XOR Gate

In this example, I trained a Neural Network (NN) to learn the logic XOR gate. I trained the NN using the stochastic gradient descent. The example is located in the file *xor-mlp.py* and the dataset in the file *XOR.dat*

## Iris UCI Data Set

In this example, I trained a NN using the dataset Iris that is provided at [UCI: Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html). This specific example validates the mini-batch gradient descent algoithm.

I tunned the NN hyperparameters in order to get successful. Therefore, if you want to save time reproducing this example you should use *eta < 0.1*. After tunning, I found out that *eta=0.05* and mini-batch size equal to 10 converges at a reasonable speed.

The source code is provided in *iris-mlp-minibatch.py* and dataset in the file *iris.data*.

### Cross-Entropy Cost Function (Switching the Neural Network Cost Function)

The source file *cross-entropy-cost-function.py* applies a NN with Cross-Entropy Cost function over the Iris Data Set. I show in this example that in order to implement a different cost functions, users must provide the NeuralNetwork construct with a class with two static methods: *fn*(Cost Functon) and *gradient* (Function Gradient).
   
```python

from lib.mlp import NeuralNetwork
 
class CrossEntropyCsst(object):
    
   @staticmethod
   def fn(a,y):
      return -1*np.sum(y*np.log(a) + 
            ((np.ones(y.shape)-y)*np.log(np.ones(a.shape)-a)))
   
   @staticmethod
   def gradient(a,y):
      return (a-y)/(a*(np.ones(a.shape)-a))

if __name__ == "__main__":
 
   nn_size = [2, 3, 1]
   mlp = NeuralNetwork(layer_size=nn_size, 
                     cost=CrossEntropyCost)

```

### MNNIST Example

I implement a Neural Network that learns to classify handwritten numbers from the MNIST dataset. I used a 3rd-party package to load the MNIST dataset into numpy arrays, which I use to train and validate the network. You'll find the source code in the file *mnist-test.py*. (Example still on going... )

The examples is still missing a more realistic training strategy, which involves splitting the data set on validation and training data sets, such as K-fold Cross validation.

### TODO
   * Regularization L1, L2, using static methods on Regularization Class.
   * Plot performance x epoch
   * Implement K-fold Cross-validation
