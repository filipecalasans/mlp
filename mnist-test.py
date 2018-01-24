import numpy as np
import lib.mlputils as mlputils

import lib.trainingutils as tu

from lib.mlp import NeuralNetwork
from lib.mlputils import RegularizationL2
from lib.mlputils import RegularizationL1
from mnist import MNIST

def train_model(nn, dataset, eta, epoches, n_folds, reg_lmbda, debug=False):
   
   performance ={"acc":[],
                 "cost":[], 
                }
   # dataset.shape[0]

   for j in range(epoches):
      cost_epoch=0
      acc_epoch=0
      kfold = tu.k_fold(n_folds, dataset_size)

      if debug:
         print("*********************************************")
         print("Epoch {}".format(j))
         print("*********************************************")

      for i in range(n_folds):
         cost, accuracy = tu.train_and_validate_kfold(nn=mlp, 
                                                      dataset=dataset,
                                                      folds=kfold, 
                                                      validation_fold=i,
                                                      eta=eta,
                                                      reg_lmbda=reg_lmbda)
         if debug:
            print("Fold[{}]: Cost={}, Accu={}".format(i, cost, accuracy))
         cost_epoch += cost
         acc_epoch += accuracy
      
      if debug:
         print("_____________________________________________")
         print(">>>  Cost:{}, Acc: {}".format(cost_epoch/n_folds, acc_epoch/n_folds))   
         print("_____________________________________________")

      performance["acc"].append(acc_epoch/n_folds)
      performance["cost"].append(cost_epoch/n_folds)

   return performance


if __name__ == "__main__":
   
   print("MLP Test using MNIST Data Set")   

   dirname = "./mnist-dataset"

   mndata = MNIST(dirname, return_type='numpy')
   training, labels = mndata.load_training()

   training_size = training.shape[0]
   input_size = training.shape[1]
   hidden_size = 100
   output_size = 10
   
   dataset = np.zeros((training_size, input_size+output_size))
   
   # Example:
   # Output Layer for class 4 = [0,0,0,0,1,0,0,0,0,0]
   # Output Layer for class 5 = [0,0,0,0,0,1,0,0,0,0]
   # Output Layer for class 0 = [1,0,0,0,0,0,0,0,0,0]
   for i in range(training_size):
      l = mlputils.get_one_zero_nn_output(output_size, labels[i])
      if l is not None:
         dataset[i] = np.append(training[i]/255, l)

   # print("======= Dataset =========\n{}".format(dataset[2]))
   
   max_col = np.amax(dataset, axis=0)
   min_col = np.amin(dataset, axis=0)

   # dataset = dataset/255.0

   # print("MAX: {}, MIN: {}".format(max_col, min_col))

   nn_size = [input_size, hidden_size, output_size]
   
   dataset_size = dataset.shape[0] 

   mlp = NeuralNetwork(layer_size=nn_size)
   
   # randomly sort the dataset. So we
   # avoid selecting alway the first examples 
   # in case we are using only a subset of dataset.
   examples_used = np.random.permutation(
               np.arange(dataset_size))

   performance_training = train_model(nn=mlp, 
                                       dataset=dataset[examples_used], 
                                       eta=0.1, 
                                       epoches=10, 
                                       n_folds=5, 
                                       reg_lmbda=0.01, 
                                       debug=True)

   print("Performance:\n{}".format(performance_training))

   print("Testing the Model trained...")

   test, test_labels = mndata.load_testing()
   dataset_test = np.zeros((test.shape[0], input_size+output_size))

   for i in range(test.shape[0]):
      tl = mlputils.get_one_zero_nn_output(output_size, test_labels[i])
      if tl is not None:
         dataset_test[i] = np.append(test[i]/255, tl)

   acc_validation = tu.validate_model(nn=mlp, dataset=dataset_test)

   print(">> Validation Dataset Accuracy: {}".format(acc_validation))
