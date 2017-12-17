import numpy as np
import math

class NeuralNetwork:

   '''   
      __init__: Initializes the weight array of matrices, the delta array of errors.
         self.w = [ W_2, W_3, ..., w_(L-1), W_L]
         self.delta = [delta_2, delta_3, ..., delta_L]

         where W_i for i in [2;L-1] is a mxm weight matrix, where m is th enumber of inputs (hidden layers)
         and W_L is a matrix nxm, where n is the number of outputs. (Output layer) 

         * We can baypass the input layer, as the input neurons output is equal to the input values.
   '''
   def __init__(self, input_size, output_size, layers=2, eta=0.1, threshold=1e-3):
      self.output_size = output_size
      self.input_size = input_size
      self.sqerror = 0
      self.threshold = threshold

      # w and beta don't contain the input layer. (This is what characterizes the NN)
      self.w = [np.random.uniform(-0.5, 0.5, (input_size, input_size)) for layer in range(layers-2)]
      self.w.append(np.random.uniform(-0.5, 0.5, (output_size, input_size)))

      self.beta = [np.random.uniform(-0.5, 0.5, (input_size, 1)) for layer in range(layers-2)]
      self.beta.append(np.random.uniform(-0.5, 0.5, (output_size, 1)))

      # a = [[x1, x2, ..., xn], [a_11, a1_2, ..., a_1n], .... [a_L1, a_L2, ..., a_Ln]]
      # a containt the input layer output.
      self.a = [np.zeros((input_size, 1)) for layer in range(layers-1)]
      self.a.append(np.zeros((output_size, 1)))

      # derivative of the activation function
      self.d_a = list(self.a)

      # delta[L] = (y - ŷ) * d_a[L]     ===> output layer Lth depth network
      # delta[l] = w[l+1]*delta[l+1] (o) d_a[l] ===> Hidden layers - 
      # where (o) is the Hadamard product
      self.delta =  list(self.a)

      print(self)


   def apply_activation_function(self, x_array):
      for xi in np.nditer(x_array, op_flags=['readwrite']):
         x_array[...] = self.activation_function(xi)

      return x_array

   def apply_d_activation_function(self, x_array):
      for xi in np.nditer(x_array, op_flags=['readwrite']):
         x_array[...] = self.d_activation_function(xi)

      return x_array

   def sigmoid(self, x):
      return 1/(1+math.exp(-x))

   def d_sigmoid(self, x):
      return self.sigmoid(x)* (1 + self.sigmoid(x))
   
   ''' If you want to change the activation 
       function you need only to change the following two functions.
       tau(zl) [activation_function] and tau'(zl) [d_activation_function].
   '''
   def activation_function(self, x):
      return self.sigmoid(x)

   def d_activation_function(self, x):
      return self.d_sigmoid(x)

   '''
      train: Test Neural Networ:
         @dataset: Numpy Matrix dataset N examples
                           [ [x1_1, x1_2, x2_3, ..., x1_n, y1],
                             [x2_1, x2_2, x2_3, ..., x2_n, y2] 
                             ... t
                             [xn_1, xn_2, xn_3, ..., xn_n, yn] ]
   '''
   def train(self, dataset):
      
      print("Training Neural Network....")

      n = dataset.shape[0]

      while True:
      
         self.sqerror = 0.0
         
         np.apply_along_axis(self.iterate_over_example, axis=1, arr=dataset)
         
         print("############################")
         print("Error: {}, Threshold: {}".format(self.sqerror/n, self.threshold))

         if (self.sqerror/n) < self.threshold:
            break
      
      print("Training done.")

   '''
      train: Apply Network over the given example, backpropagate the error and update the 
             weights.
         @example: Numpy Array containing one training example
                         [x1, x1, x2, ..., xn, y]
   '''
   def iterate_over_example(self, example):
      
      col = example.shape[0]

      if col is not (output_size + input_size): 
         return

      # print("# col: {}".format(col))

      # [x1, x2, ..., xn, 1]
      x = example[:(col-output_size)].reshape(input_size, 1)
      y = example[(col-output_size):].reshape(output_size, 1)

      self.update_neuron_outputs(x,y)

      # calculate the error in the output layer
      g = self.cost_function_gradient(y, self.a[len(self.a)-1])

      self.delta[len(self.delta)-1] = g * self.d_a[len(self.d_a)-1]

      print("============================================")
      print("Network Status")
      print("============================================")

      print("X = {}, Y = {}".format(x, y))
      print("A = {}".format(self.a))
      print(self)
      print("++++++++++++++++++++++++++++++++++++++++++++")
      print(self.network_status())
      print("============================================")


   def update_neuron_outputs(self, x, y):
      
      '''
         A[l] = activation(Z[l])
         Z[l] = W[l] * A[l-1] + beta[l]
      
         *Inputs and outputs are handle as vertical matrices. (i.e)
         X = transpose(<x1, x2, ... xn>)
         Y = transpose(<y1, y2, ..., ym>)
         
         resulting on neuron output vertical matrices (i.e)
         A = transpose(<a1, a2, ..., an>) for each hidden layer
         Z = transpose<z1, z2, ..., zn>) for each hidden layer
         DELTA = transpose<delta_1, delta_2, ..., delta_n>)

         A = transpose(<a1, a2, ..., am>) for each output layer
         Z = transpose<z1, z2, ..., zm>) for each output layer
         DELTA = transpose<delta_1, delta_2, ..., delta_m>)
      '''
      self.a[0] = x

      for layer, w_i in enumerate(self.w):
         
         # self.w do not consider the input layer, so add 1.
         # We want to update the neuron ouput of the layer(l),
         # so we calculate the activation output using the inputs from the (l-1)
         # and beta from the lth layer.
         output_index = layer + 1
         print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
         print("Calculating Layer {}".format(layer))
         print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

         # apply weights and beta on the previous layer output values. 
         print("A = {}".format(self.a[layer]))
         print("W = {}".format(w_i))

         # *** beta is indexed as W - we do not consider the w 
         # from the input layer (Indentity Matrix) neither the beta
         # from the input layer.
         z_l = np.matmul(w_i,self.a[layer]) + self.beta[layer] 
         
         # apply activation function (generally sigmoide)
         print("zl = {}".format(z_l))
         
         self.a[output_index] = self.apply_activation_function(z_l)

         self.d_a[output_index] = self.apply_d_activation_function(z_l)


   def cost_function_gradient(self, y, output):
      return y - output


   '''
      classify: apply the neural network over the given input
         @x: numpy array 
               [x1, x2, ... , xn]
   '''
   def classify(self, x):
      pass  

   def __str__(self):
      return "Weights:\n{}\Beta:\n{}".format(self.w, self.beta)

   def network_status(self):
      return ("DELTA: {}\n".format(self.delta) + 
         "tau(z): {}\n".format(self.a) +
         "tau'(z): {}\n".format(self.d_a))


if __name__ == "__main__":
   print("MLP Test")   

   filename = "XOR.dat"

   '''
      @dataset: array of arrays
               [  [x1, x1, x2, ..., xn, y],
                  [x1, x1, x2, ..., xn, y], 
                  [x1, x1, x2, ..., xn, y] ]
   '''
   dataset = np.loadtxt(open(filename, "rb"), delimiter=" ")
  
   print("DataSet: {}".format(dataset))
   input_size = dataset.shape[1] - 1 

   print("INPUT SIZE {}".format(input_size))

   output_size = 1
   layers = 3 # input, hidden and output layer

   mlp = NeuralNetwork(input_size, output_size, layers)
   
   mlp.train(dataset)

   print(mlp)