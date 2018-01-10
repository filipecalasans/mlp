from lib.mlp import NeuralNetwork
import numpy as np

if __name__ == "__main__":
   
   print("MLP Test usin XOR gate example")   

   filename = "iris.data"

   dataset = np.loadtxt(open(filename, "rb"), delimiter=",")

   output_size = 3
   input_size = dataset.shape[1] - output_size

   print("======= Dataset =========\n{}".format(dataset))
   
   max_col = np.amax(dataset, axis=0)
   min_col = np.amin(dataset, axis=0)

   dataset = (dataset-min_col)/(max_col - min_col)

   print("MAX: {}, MIN: {}".format(max_col, min_col))

   nn_size = [input_size, 3, output_size]

   mlp = NeuralNetwork(nn_size, True)

   batch_size = 10

   # 
   #  Implement cross-entropy cost function: (Sum for each output neuron)
   #  C = -[yln(a) + (1-y)ln(1-a)]
   #  gradient C = (a-y)/(a(1-a)) 
   def cost_entropy_gradient(a,y):
      return (a-y)/(a*(np.ones(a.shape)-a))

   def cross_entropy(a,y):
      return -1*np.sum(y*np.log(a) + ((np.ones(y.shape)-y)*np.log(np.ones(a.shape)-a)))
   
   mlp.cost_function = cross_entropy 
   mlp.cost_function_gradient = cost_entropy_gradient   

   mlp.train(dataset, eta=0.05, threshold=1e-1)

   a, y = mlp.classify(dataset[63][0:input_size])
   print("Y: {}, Ŷ: {}, round(Ŷ): {}".format(dataset[63][-(input_size-1):].flatten(), \
            y.flatten(), np.round(y).flatten()))

   a, y = mlp.classify(dataset[0][0:input_size])
   print("Y: {}, Ŷ: {}, round(Ŷ): {}".format(dataset[0][-(input_size-1):].flatten(), \
            y.flatten(), np.round(y).flatten()))

   a, y = mlp.classify(dataset[110][0:input_size])
   print("Y: {}, Ŷ: {}, round(Ŷ): {}".format(dataset[110][-(input_size-1):].flatten(), \
            y.flatten(), np.round(y).flatten()))