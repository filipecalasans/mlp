import numpy as np
import math

class NeuralNetwork:

   w_init_max = 0.5
   w_init_min = -0.5

   b_init_max = 0.5
   b_init_min = -0.5

   '''   
      __init__: Initializes the weight array of matrices, the delta array of errors.
         self.w = [ W_2, W_3, ..., w_(L-1), W_L]
         self.delta = [delta_2, delta_3, ..., delta_L]

         where W_i for i in [2;L-1] is a mxm weight matrix, where m is th enumber of inputs (hidden layers)
         and W_L is a matrix nxm, where n is the number of outputs. (Output layer) 

         * We can baypass the input layer, as the input neurons output is equal to the input values.
   '''
   def __init__(self, layer_size=[2,3,1], eta=0.1, threshold=1e-3):
      
      self.layer_size = layer_size
      self.sqerror = 0
      self.threshold = threshold
      self.eta = eta

      self.w = list()
      self.beta = list()
      self.a = list()
      self.d_a = list()
      self.delta = list()
      
      if(type(layer_size) is not list):
         print("layer_size must be an array describing the Network Size and Depth")
         exit()

      if(len(layer_size) < 3):
         print("layer_size doesn't describe an MLP. len(layer_size) < 3 ")
         exit()

      for l in range(len(layer_size)-1):
         # w and beta don't contain the input layer. (These matrices characterize the NN)
         # print((layer_size[l+1], layer_size[l]))
         self.w.append(np.random.uniform(self.w_init_min, self.w_init_max, (layer_size[l+1], layer_size[l])))
         self.beta.append(np.random.uniform(self.b_init_min, self.b_init_max, (layer_size[l+1], 1)))

      print(self.w)
      print(self.beta)

      for l in range(0, len(layer_size)):
         # a = [[x1, x2, ..., xn].T, [a_11, a1_2, ..., a_1n].T, .... [a_L1, a_L2, ..., a_Ln].T]
         # vector list contains the input layer output vector, which is equal to the 
         # input vector for a given example.
         self.a.append(np.zeros((layer_size[l], 1)))

      print("A: {}".format(self.a))
      # derivative of the activation function
      self.d_a = list(self.a)
      print("d_A: {}".format(self.d_a))
      # delta[L] = (y - ŷ) * d_a[L]     ===> for the output layer in a L-depth network
      # delta[l] = w[l+1]*delta[l+1] (o) d_a[l] ===> for the Hidden layers
      # where (o) is the Hadamard product
      self.delta =  list(self.a)
      print("delta: {}".format(self.delta))

   # Hadamard Product of the activation function (Tau) over the
   # net vector. net vector is the vector Z[l]
   # where Z[l] = W[l] * A[l-1] + beta[l]
   # Activation function A[L].
   # Where A = Tau(Z)
   def apply_activation_function(self, x_array):
      fnet = np.vectorize(self.activation_function)
      y = fnet(x_array)
      return y 
   
   '''
   Hadamard Product of the derivative activation function (Tau') over the
   net vector. A[L]
   
   net vector is the vector Z[l]
   where Z[l] = W[l] * A[l-1] + beta[l]
   
    Where A = Tau(Z)
    d_A = Tau'(Z)
   '''
   def apply_d_activation_function(self, x_array):
      d_fnet = np.vectorize(self.d_activation_function)
      y = d_fnet(x_array)
      return y
      
   
   '''
      Default activation function 
   '''
   def sigmoid(self, x):
      return 1/(1+math.exp(-x))
   
   '''
      Default activation function derivative 
   '''
   def d_sigmoid(self, x):
      return self.sigmoid(x)*(1 - self.sigmoid(x))
   
   ''' If you want to change the activation 
       function, you need only to change the following two functions.
       tau(zl) [activation_function] and tau'(zl) [d_activation_function].

       @x: float value.

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
   def train(self, dataset, max_iterations=0):
      
      print("Training Neural Network....")

      n = dataset.shape[0]

      count = 0

      while True:
      
         self.sqerror = 0.0
         count += 1

         np.apply_along_axis(self.iterate_over_example, axis=1, arr=dataset)
         
         print(self.training_status(n))
         print("##############################################")
         
         if ((self.sqerror/n) < self.threshold) or (count>max_iterations and max_iterations>0):
            break
      
      print("##############################################")
      print(self.training_status(n))
      print("##############################################")
         
      print("Training done.")

   '''
      train: Apply Network over the given example, backpropagate the error and update the 
             weights.
         @example: Numpy Array containing one training example
                         [x1, x1, x2, ..., xn, y]
   '''
   def iterate_over_example(self, example):
      
      col = example.shape[0]

      output_size =  self.layer_size[len(self.layer_size)-1]
      input_size = self.layer_size[0]

      # print("output_size {}".format(output_size))

      if col is not (input_size + output_size): 
         return

      # print("# col: {}".format(col))
      # print("+++++++++++++++++ Before +++++++++++++++++++")
      # print(self)
      # print("++++++++++++++++++++++++++++++++++++++++++++")
      # transpose([x1, x2, ..., xn, 1])
      x = example[:(col-output_size)].reshape(input_size, 1)
      y = example[(col-output_size):].reshape(output_size, 1)

      self.update_neuron_outputs(x)

      # Update the error matrices
      self.update_error(y)

      # backpropagate the error through the network.
      self.backpropagate()

      self.apply_learning_equation()

      # print("============================================")
      # print("Network Status")
      # print("============================================")

      # print("X = {}, Y = {}".format(x, y))
      # print(self)
      # print("++++++++++++++++++ After +++++++++++++++++++")
      # print(self.network_status())
      # print("++++++++++++++++++++++++++++++++++++++++++++")

   def update_neuron_outputs(self, x):
      
      '''
         for each layer we have: 

         A[l] = Tau(Z[l])
         Z[l] = W[l] * A[l-1] + beta[l]

         where Tau is the activation function used.

         *Inputs and outputs are handle as vertical matrices. (i.e)
         X = transpose(<x1, x2, ... xn>)
         Y = transpose(<y1, y2, ..., ym>)
         
         resulting on neuron output vertical matrices (i.e)
         A = transpose(<a1, a2, ..., an>) for each hidden layer
         Z = transpose<z1, z2, ..., zn>) for each hidden layer
         DELTA = transpose<delta_1, delta_2, ..., delta_n>)

         A = transpose(<y1, y2, ..., ym>) for each output layer
         Z = transpose<z1, z2, ..., zm>) for each output layer
         DELTA = transpose<delta_1, delta_2, ..., delta_m>)

         remembering that A[0] = X = transpose(<x1, x2, ..., xn>) for input layer.
      '''
      self.a[0] = x

      for layer, w_i in enumerate(self.w):
         
         # self.w doesn't consider the input layer, so add 1.
         # We want to update the neuron ouput of the layer(l),
         # so we calculate the activation output using inputs from (l-1) layer
         # and beta from the lth layer.
         output_index = layer + 1
        
         # print("layer: {}, dim(wi): {}, dim(a): {}, dim(beta): {}".format(layer, w_i.shape, self.a[layer].shape, self.beta[layer].shape))

         # *** beta is indexed as W - we do not consider the w 
         # from the input layer (Indentity Matrix) neither the beta
         # from the input layer.
         z = np.matmul(w_i, self.a[layer]) + self.beta[layer] 
         
         # apply activation function (default is sigmoide)  
         # print("Z[{}]: {}".format(output_index, self.a[output_index])) 
         self.a[output_index] = self.apply_activation_function(z)
         # print("A[{}]: {}".format(output_index, self.a[output_index]))
         self.d_a[output_index] = self.apply_d_activation_function(z)


   def update_error(self, y):
      
      '''
         Error in the output layer (Lth layer) is given by:
            delta[L] = (Y - Ŷ) * A'(Z[L]) or
            delta[L] = (Y - Ŷ) * Tau'(Z[L])

         Error in the hidden layers are:
            delta[l] = (w[l+1] * delta[l+1]) (o) Tau'(Z[l]) or 
            delta[l] = (w[l+1] * delta[l+1]) (o) A'(Z[l])
            
            where,  (o) is the Hadamard Product operator [apply multiplication element wise].
                 , A' = Tau' (samething) 
      '''
      error = (y - self.a[-1])

      # calculate the error in the output layer 
      # Apply activation function derivative using hadamard product operation
      self.delta[-1] = error * self.d_a[-1] 
      self.sqerror += np.sum(error**2)


   def backpropagate(self):
      
      output_layer = len(self.w)-1
      
      # loop from [output_layer-1 ... 0]
      # Remember Layer 0 in the W array is the first hidden layer
      # print(list(range(output_layer, -1, -1)))

      for l in range(output_layer, 0, -1):
         # print("backpropagate layer {}".format(l))
         # print("w[{}]: {}, delta[{}]: {}, d_a[{}]: {}".format(l+1, self.w[l+1], l+2, self.delta[l+2], l+1, self.d_a[l+1]))
         self.delta[l] = np.matmul(self.w[l].T, self.delta[l+1])*self.d_a[l]
         # w[0] * delta[1]

   def apply_learning_equation(self):
      '''
         This is the implementation of the Gradient Descent Algorithm.

         for each layer in the hidden layer:
            W(t+1) = W(t) + eta * delta[l] * A[l-1].T
      '''
      output_layer = len(self.w)-1

      # loop from [output_layer ... 1]
      # Remember Layer 0 in the W array is the first hidden layer
      for l in range(output_layer, -1, -1):
         # print("Learning Equeation Layer {}".format(l))
         # print("w[{}]: {}, delta[{}]: {}, a[{}]: {}".format(l, self.w, l+1, self.delta[l+1], l+1, self.a[l+1]))
         self.w[l] = self.w[l] + self.eta*np.matmul(self.delta[l+1],self.a[l].T)         
         self.beta[l] = self.beta[l] + self.eta*self.delta[l+1]

   '''
      classify: apply the neural network over the given input
         @x: numpy array 
               [x1, x2, ... , xn]
   '''
   def classify(self, x):
      
      input_size = self.layer_size[0]
      
      if(x.shape[0] is not input_size): 
         return None, None

      x = x.reshape(input_size, 1)
      
      outputs = [] # TODO: Optimize
      outputs.append(x)

      for w_index, w_i in enumerate(self.w):
         
         # self.w doesn't consider the input layer, so add 1.
         # We want to update the neuron ouput of the layer(l),
         # so we calculate the activation output using inputs from (l-1) layer
         # and beta from the lth layer.
         output_index = w_index + 1

         # *** beta is indexed as W - we do not consider the w 
         # from the input layer (Indentity Matrix) neither the beta
         # from the input layer.
         z = np.matmul(w_i, outputs[output_index-1]) + self.beta[w_index]          
         outputs.append(self.apply_activation_function(z))

      return outputs, outputs[len(outputs)-1]

   def __str__(self):
      return "Weights:\n{}\nBeta:\n{}".format(self.w, self.beta)


   def network_status(self):
      return ("A: {}\n".format(self.a) +
              "A': {}\n".format(self.d_a) + 
         "DELTA:\n{}\n".format(self.delta) +
         "W:\n{}\n".format(self.w) +
         "Beta:\n{}\n".format(self.beta))

   def training_status(self, n):
      return "Error: {} / {}".format(self.sqerror/n, self.threshold)



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
  
   input_size = dataset.shape[1] - 1
   output_size = 1

   nn_size = [input_size, 2, output_size]

   print("DataSet: {}".format(dataset))
   print("NN SIZE {}".format(nn_size))

   mlp = NeuralNetwork(nn_size, eta=0.1, threshold=1e-3)
         
   # print(mlp)
   
   mlp.train(dataset, max_iterations=100000)

   outputs, output = mlp.classify(np.array([0,0]))
   
   print(mlp)

   x = np.array([0,0])
   outputs, output = mlp.classify(x)
   print("==========================")
   # print("Z: {}".format(outputs))
   print("x: {}, ŷ: {}".format(x, output))

   x = np.array([0,1])
   outputs, output = mlp.classify(x)
   print("==========================")
   # print("Z: {}".format(outputs))
   print("x: {}, ŷ: {}".format(x, output))

   x = np.array([1,0])
   outputs, output = mlp.classify(x)
   print("==========================")
   # print("Z: {}".format(outputs))
   print("x: {}, ŷ: {}".format(x, output))


   x = np.array([1,1])
   outputs, output = mlp.classify(x)
   print("==========================")
   # print("Z: {}".format(outputs))
   print("x: {}, ŷ: {}".format(x, output))

