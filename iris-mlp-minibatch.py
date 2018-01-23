from lib.mlp import NeuralNetwork
import numpy as np

if __name__ == "__main__":
   
   print("MLP Test using IRIS Data Set")   

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

   mlp = NeuralNetwork(layer_size=nn_size, debug_string=True)

   batch_size = 10

   mlp.train_batch(dataset, batch_size=batch_size, eta=0.05, threshold=1e-3)
   # mlp.train(dataset, eta=0.05, threshold=1e-3)

   a, y = mlp.classify(dataset[63][0:input_size])
   print("Y: {}, Ŷ: {}".format(dataset[63][-(input_size-1):], np.round(y)))

   a, y = mlp.classify(dataset[0][0:input_size])
   print("Y: {}, Ŷ: {}".format(dataset[0][-(input_size-1):], np.round(y)))

   a, y = mlp.classify(dataset[110][0:input_size])
   print("Y: {}, Ŷ: {}".format(dataset[110][-(input_size-1):], np.round(y)))