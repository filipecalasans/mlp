import numpy as np

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
      self.w = [np.random.uniform(-0.5, 0.5, (input_size, input_size)) for layer in range(layers-2)]
      self.w.append(np.random.uniform(-0.5, 0.5, (output_size, input_size)))
      self.delta = [np.zeros(input_size) for layer in range(layers-2)]
      self.delta.append(np.zeros(output_size))
      self.sqerror = 0
      
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

      # print("# col: {}".format(col))

      # [x1, x2, ..., xn, 1]
      x = np.append(example[:(col-1)], 1)
      y = example[(col-1):]

      print("X = {}, y = {}".format(x, y))

   '''
      classify: apply the neural network over the given input
         @x: numpy array 
               [x1, x2, ... , xn]
   '''
   def classify(self, x):
      pass  

   def __str__(self):
      return "Weights:\n{}\nDelta:\n{}".format(self.w, self.delta)


if __name__ == "__main__":
   print("MLP Test")   
   input_size = 3
   output_size = 1
   layers = 3 # input, hidden and output layer
   n = NeuralNetwork(input_size, output_size, layers)
   print(n)