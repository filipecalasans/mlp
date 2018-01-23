import numpy as np
import lib.mlputils as mlputils

import lib.trainingutils as tu

from lib.mlp import NeuralNetwork
from lib.mlputils import RegularizationL2
from lib.mlputils import RegularizationL1
from mnist import MNIST


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
      l = mlputils.get_one_zero_nn_output(output_size, labels[i])
      if l is not None:
         data_set[i] = np.append(training[i]/255, l)

   # print("======= Dataset =========\n{}".format(data_set[2]))
   
   max_col = np.amax(data_set, axis=0)
   min_col = np.amin(data_set, axis=0)

   # data_set = data_set/255.0

   # print("MAX: {}, MIN: {}".format(max_col, min_col))

   nn_size = [input_size, hidden_size, output_size]

   mlp = NeuralNetwork(layer_size=nn_size)

   batch_size = 10

   # mlp.train_batch(data_set[:500], batch_size=batch_size, eta=0.05, threshold=1e-3)
   # mlp.train(data_set[:500], eta=0.1, threshold=1e-2)  

   n_folds = 10
   
   epoches=10
   performance ={"acc":[],
                 "cost":[], 
                 }

   for j in range(epoches):
      cost_epoch=0
      acc_epoch=0
      kfold = tu.k_fold(n_folds, 1000)
      print("*********************************************")
      print("Epoch {}".format(j))
      print("*********************************************")
      for i in range(n_folds):
         cost, accuracy = tu.train_and_validate_kfold(nn=mlp, 
                                                      dataset=data_set[:1000],
                                                      folds=kfold, 
                                                      validation_fold=i,
                                                      eta=0.1,
                                                      reg_lmbda=0.01)
         print("Fold[{}]: Cost={}, Accu={}".format(i, cost, accuracy))
         cost_epoch += cost
         acc_epoch += accuracy

      print("_____________________________________________")
      print(">>>  Cost:{}, Acc: {}".format(cost_epoch/n_folds, acc_epoch/n_folds))   
      print("_____________________________________________")

      performance["acc"].append(acc_epoch/n_folds)
      performance["cost"].append(cost_epoch/n_folds)

   print("Performance:\n{}".format(performance))