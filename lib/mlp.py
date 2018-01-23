import numpy as np
import math
import lib.mlputils as mlputils

'''

'''
class NeuralNetwork(object):

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

         * We can baypass the input layer.
   '''
   def __init__(self, 
               layer_size=[2,3,1], 
               activation=mlputils.SigmoidActivation,
               cost=mlputils.QuadraticCost,
               regularization=mlputils.RegularizationNone,
               debug_string=False):

      self.activation_function=activation
      self.is_debug = debug_string
      self.layer_size = layer_size
      self.sqerror = 0
      self.threshold = 1e-3
      self.eta = 0.1
      self.count = 0
      self.cost = cost
      self.regularization=regularization
      self.reg_lmbda = 0.1

      self.batch_size = 1

      self.w = list()
      self.beta = list()
      self.a = list()
      self.d_a = list()
      self.delta = list()
      self.nabla_w = list()
      self.nabla_beta = list()

      if(type(layer_size) is not list):
         print("layer_size must be an array describing the Network Size and Depth")
         exit()

      if(len(layer_size) < 3):
         print("layer_size doesn't describe a MLP. len(layer_size) < 3 ")
         exit()

      for l in range(len(layer_size)-1):
         # w and beta don't contain the input layer. (These matrices characterize the NN)
         # print((layer_size[l+1], layer_size[l]))
         input_size = layer_size[0]
         self.w.append(np.random.normal(0, 1/math.sqrt(input_size), (layer_size[l+1], layer_size[l])))
         self.beta.append(np.random.uniform(self.b_init_min, self.b_init_max, (layer_size[l+1], 1)))

         self.nabla_w.append(np.zeros(self.w[l].shape))
         self.nabla_beta.append(np.zeros((self.beta[l].shape)))

      self.n_weights=0.0
      for wi in self.w:
          self.n_weights += wi.shape[0]*wi.shape[1] 

      # if self.is_debug:
      #    print("W:\n{}".format(self.w))
      #    print("Beta:\n{}".format(self.beta))
      #    print("nabla_w:\n{}".format(self.nabla_w))
      #    print("nabla_beta:\n{}".format(self.nabla_beta))

      for l in range(0, len(layer_size)):
         # a = [[x1, x2, ..., xn].T, [a_11, a1_2, ..., a_1n].T, .... [a_L1, a_L2, ..., a_Ln].T]
         # vector list contains the input layer output vector, which is equal to the 
         # input vector for a given example.
         self.a.append(np.zeros((layer_size[l], 1)))
      
      # if self.is_debug:
      #    print("A: {}".format(self.a))
      
      # derivative of the activation function
      self.d_a = list(self.a)
      
      # if self.is_debug:
      #    print("d_A: {}".format(self.d_a))
      
      # delta[L] = (y - ŷ) * d_a[L]     ===> for the output layer in a L-depth network
      # delta[l] = w[l+1]*delta[l+1] (o) d_a[l] ===> for the Hidden layers
      # where (o) is the Hadamard product
      self.delta =  list(self.a)
      
      # if self.is_debug:
      #    print("delta: {}".format(self.delta))

   # Hadamard Product of the activation function (Tau) over the
   # net vector. net vector is the vector Z[l]
   # where Z[l] = W[l] * A[l-1] + beta[l]
   # Activation function A[L].
   # Where A = Tau(Z)
   def apply_activation_function(self, x_array):
      fnet = np.vectorize((self.activation_function).f)
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
      d_fnet = np.vectorize((self.activation_function).df)
      y = d_fnet(x_array)
      return y
         
   '''
      train: Test Neural Networ: (Stochastic)
         @dataset: Numpy Matrix dataset N examples
                           [ [x1_1, x1_2, x2_3, ..., x1_n, y1],
                             [x2_1, x2_2, x2_3, ..., x2_n, y2] 
                             ... t
                             [xn_1, xn_2, xn_3, ..., xn_n, yn] ]
   '''
   def train(self, dataset, eta=0.1, threshold=1e-3, reg_lmbda = 0.01, max_iterations=0):
      
      if self.is_debug:
         print("Training Neural Network ...")
      
      n = dataset.shape[0]
      
      self.count = 0
      self.eta = eta
      self.threshold = threshold
      self.reg_lmbda = reg_lmbda
      
      epochs = 0

      while True:
      
         self.sqerror = 0.0
         self.count += 1

         np.apply_along_axis(self.iterate_over_example, axis=1, arr=dataset)
         
         if self.is_debug:
            print(self.training_status(n))
            print("##############################################")
         
         if ((self.sqerror/n) < self.threshold) or (self.count>max_iterations and max_iterations>0):
            break
      
      if self.is_debug:
         print(self.training_status(n))
         print("##############################################")
         print("Training is done.")

   def update_batch(self, batch, eta=0.1):

      self.eta = eta
      # self.sqerror = 0.0
      self.batch_size = batch.shape[0]

      for example in batch:
         self.count += 1
         self.forward_and_backpropagate(example)
         self.accumulate_and_apply_learning()

   def train_batch(self, dataset, batch_size, eta=0.1,  reg_lmbda=0.01, threshold=1e-3, max_iterations=0):
      
      if dataset.shape[0]%batch_size is not 0:
         print("[ERROR] len(dataset) % batch_size != 0")
         return
      
      self.threshold = threshold
      self.reg_lmbda = reg_lmbda

      index = np.array(range(dataset.shape[0]))
      
      batch_init = 0
      n = dataset.shape[0]

      while True:
         
         batch_dataset = np.array(dataset[batch_init:(batch_init+batch_size)])
         self.update_batch(batch_dataset, eta)
         batch_init = (batch_init + batch_size) % dataset.shape[0]

         if ((self.sqerror/n) < self.threshold) or (self.count>max_iterations and max_iterations>0):
            print(self.training_status(n))
            print("Trainig done.")
            print("##############################################")
            return
         
         if self.count % n == 0:
            if self.is_debug:
               print(self.training_status(n))
               print("##############################################")
            self.sqerror = 0

   '''
      train: Apply Network over the given example, backpropagate the error and update the 
             weights.
         @example: Numpy Array containing one training example
                         [x1, x1, x2, ..., xn, y]
   '''
   def iterate_over_example(self, example):
      self.forward_and_backpropagate(example)
      self.apply_learning_equation()

   def iterate_over_batch_example(self, example):
      self.forward_and_backpropagate(example)
      self.accumulate_and_apply_learning()

   def forward_and_backpropagate(self, example):

      col = example.shape[0]

      output_size =  self.layer_size[len(self.layer_size)-1]
      input_size = self.layer_size[0]

      if col != (input_size + output_size): 
         print("Error col != input_size + output_size. [{} != {} + {}], {}".format(
               col, input_size, output_size, input_size+output_size))
         return

      x = example[:(col-output_size)].reshape(input_size, 1)
      y = example[(col-output_size):].reshape(output_size, 1)
      
      self.update_neuron_outputs(x)

      # Update the error matrices
      self.update_error_out_layer(y)

      # backpropagate the error through the network.
      self.backpropagate()

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

         # *** beta is indexed as W - we do not consider the
         # input layer weights (Indentity Matrix) neither the beta
         # from the input layer.
         z = np.matmul(w_i, self.a[layer]) + self.beta[layer] 
         
         # apply activation function (default is sigmoide)  
         self.a[output_index] = self.apply_activation_function(z)
         self.d_a[output_index] = self.apply_d_activation_function(z)


   def update_error_out_layer(self, y):
      
      '''
         Error in the output layer (Lth layer) is given by:
            delta[L] = (Y - Ŷ) * A'(Z[L]) or
            delta[L] = (Y - Ŷ) * Tau'(Z[L])
            
            where,  (o) is the Hadamard Product operator [apply multiplication element wise].
                 , A' = Tau' (samething) 
      '''
      gradient = (self.cost).gradient(self.a[-1], y)

      # calculate the error in the output layer 
      # Apply activation function derivative using hadamard product operation
      self.delta[-1] = gradient * self.d_a[-1] 
      self.sqerror += ((self.cost).fn(self.a[-1], y) + 
                        self.regularization.fn(self.reg_lmbda, self.n_weights, self.w))

   def backpropagate(self):
      '''
         Error in the hidden layers are:
         delta[l] = (w[l+1].T * delta[l+1]) (o) Tau'(Z[l]) or 
         delta[l] = (w[l+1].T * delta[l+1]) (o) A'(Z[l])
         
         where,  (o) is the Hadamard Product operator [apply multiplication element wise].
               , A' = Tau' (samething) 
      '''
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
         Stochastic Gradient Descent Algorithm.

         for each layer in the hidden layer:
            W(t+1) = W(t) - eta * delta[l] * A[l-1].T
      '''
      
      
      output_layer = len(self.w)-1
      d_regularization = self.regularization.df(self.reg_lmbda, self.n_weights, self.w)

      # loop from [output_layer ... 1]
      # Remember Layer 0 in the W array is the first hidden layer
      for l in range(output_layer, -1, -1):  
         n_w, n_beta = self.calculate_update_step(l)

         # print("Nabla_w: \n{}\n".format(n_w))
         # print("Nabla_beta: \n{}\n".format(n_beta))

         self.w[l] = self.w[l] - self.eta*n_w - self.eta*d_regularization[l]       
         self.beta[l] = self.beta[l] - self.eta*n_beta

   def accumulate_and_apply_learning(self):
      '''
         accumulate the gradient descent and increment step throught batch examples.
         Apply the minibatch gradient descent equation at the end of the batch iteration.

         implementation of batch/minibatch Gradient Descent Algorithm.

         for each layer in the hidden layer:
            W(t+1) = W(t) - (eta/batch_size) * delta[l] * A[l-1].T
      '''
      output_layer = len(self.w)-1
      d_regularization = self.regularization.df(self.reg_lmbda, self.n_weights, self.w)
      # loop from [output_layer ... 1]
      # Remember: Layer 0 in the W array is the first hidden layer
      for l in range(output_layer, -1, -1):  
         
         n_w, n_beta = self.calculate_update_step(l)
         
         self.nabla_w[l] = self.nabla_w[l] + n_w
         self.nabla_beta[l] = self.nabla_beta[l] + n_beta

         # print("Nabla_w: \n{}\n".format(self.nabla_w[l]))
         # print("Nabla_beta: \n{}\n".format(self.nabla_beta[l]))

         if self.count%self.batch_size == 0:
            # print("COUNT: {}, LAYER: {}".format(self.count, l))
            self.w[l] = (self.w[l] - ((self.eta/self.batch_size) * self.nabla_w[l]) - 
                        ((self.eta/self.batch_size)*d_regularization[l]))         
            self.beta[l] = self.beta[l] - ((self.eta/self.batch_size)*self.nabla_beta[l])
            
            self.nabla_w[l] = np.zeros(self.w[l].shape)
            self.nabla_beta[l] = np.zeros(self.beta[l].shape)
            

   def calculate_update_step(self, l):
      n_w = np.matmul(self.delta[l+1],self.a[l].T)
      n_beta = self.delta[l+1]
      return n_w, n_beta

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

         # *** beta is indexed as W - we do not consider the weights
         # from the input layer (Indentity Matrix) neither the beta
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



