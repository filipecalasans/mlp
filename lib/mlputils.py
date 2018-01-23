import numpy as np
import math

#### Miscellaneous functions and Classes
def f_sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+math.exp(-x))

def df_sigmoid(x):
    """Derivative of the sigmoid function."""
    return f_sigmoid(x)*(1-f_sigmoid(x))

class SigmoidActivation(object):
   '''
      Sigmoid Activation Function. This's the default activation function.
   '''
   @staticmethod
   def f(x):
      return f_sigmoid(x)

   @staticmethod
   def df(x):
      return df_sigmoid(x)
   
class QuadraticCost(object):
   @staticmethod
   def fn(a, y):
      error = (a-y)
      return 0.5*np.sum(error**2)

   @staticmethod
   def gradient(a, y):
      return (a-y)

class CrossEntropyCost(object):
   #  Implement cross-entropy cost function: (Sum for each output neuron)
   #  C = -sum([yln(a) + (1-y)ln(1-a)])
   @staticmethod
   def fn(a,y):
      return -1*np.sum(y*np.log(a) + 
            ((np.ones(y.shape)-y)*np.log(np.ones(a.shape)-a)))

   #gradient(C) = (a-y)/(a(1-a))
   @staticmethod
   def gradient(a,y):
      return (a-y)/(a*(np.ones(a.shape)-a))

'''
On Regularization methods:
   @w: List of NumpyArrays representing the multilayer network
   @n: Number of weights in the Network
'''
class RegularizationNone(object):
   @staticmethod
   def fn(lmbda, n, w):
      return 0
   
   @staticmethod
   def df(lmbda, n, w):
      return [ np.zeros(wi.shape) for wi in w]

class RegularizationL1(object):
   @staticmethod
   def fn(lmbda, n, w):
      w_sum = 0
      for wi in w:
         w_sum += np.sum(wi)

      return (lmbda*w_sum)/n
   
   @staticmethod
   def df(lmbda, n, w):
      return [ (lmbda*np.sign(wi))/n for wi in w] 

class RegularizationL2(object):
   @staticmethod
   def fn(lmbda, n, w):
      w_sum = 0
      for wi in w:
         w_sum += np.sum(wi**2)

      return (0.5*lmbda*w_sum)/n
   
   @staticmethod
   def df(lmbda, n, w):
      return [ (lmbda*wi)/n for wi in w]
      
