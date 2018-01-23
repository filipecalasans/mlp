from lib.mlp import NeuralNetwork
import numpy as np

if __name__ == "__main__":

   print("MLP Test usin XOR gate")   

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

   mlp = NeuralNetwork(layer_size=nn_size, debug_string=True)
   
   mlp.train(dataset, eta=0.1, threshold=1e-3, max_iterations=100000)

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
