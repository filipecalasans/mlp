from lib.mlp import NeuralNetwork
from lib.mlputils import RegularizationL2
from lib.mlputils import RegularizationL1

import numpy as np
from mnist import MNIST

def get_one_zero_nn_output(output_size, class_number):
   out = np.zeros(output_size)
   if class_number-1 > output_size:
      return None

   out[class_number-1] = 1
   return out

if __name__ == "__main__":
   
   print("MLP Test using MNIST Data Set")   

   dirname = "./mnist-dataset"

   mndata = MNIST(dirname, return_type='numpy')
   training, labels = mndata.load_training()
   test = mndata.load_testing()

   training_size = training.shape[0]
   input_size = training.shape[1]
   hidden_size = 100
   output_size = 10
   
   data_set = np.zeros((training_size, input_size+output_size))

   for i in range(training_size):
      l = get_one_zero_nn_output(output_size, labels[i])
      if l is not None:
         data_set[i] = np.append(training[i]/255, l)

   # print("======= Dataset =========\n{}".format(data_set[2]))
   
   max_col = np.amax(data_set, axis=0)
   min_col = np.amin(data_set, axis=0)

   # data_set = data_set/255.0

   # print("MAX: {}, MIN: {}".format(max_col, min_col))

   nn_size = [input_size, hidden_size, output_size]

   mlp = NeuralNetwork(layer_size=nn_size, debug_string=True)

   batch_size = 10

   # mlp.train_batch(data_set[:500], batch_size=batch_size, eta=0.05, threshold=1e-3)
   mlp.train(data_set[:500], eta=0.1, threshold=1e-2)