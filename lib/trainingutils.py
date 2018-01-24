import numpy as np 
from lib.mlp import NeuralNetwork

def k_fold(k, dataset_size):
   if k > dataset_size:
      return []

   permutation = np.random.permutation(
                     np.arange(dataset_size))
   
   step = round(dataset_size/k)
   folds = [permutation[i:i+step] for i in range(0, dataset_size, step)]
   return folds

def train_and_validate_kfold(nn, dataset, folds, validation_fold, 
                             eta=0.1, reg_lmbda=0.01):
   cost = 0
   print("=========== Validation Fold # {} ===============".format(validation_fold))

   for i,fold in enumerate(folds):
      if i != validation_fold:
         # print("Trainig Fold[{}]".format(i))
         cost += nn.train(dataset[fold], eta=eta, threshold=-1, reg_lmbda=reg_lmbda, max_iterations=1)   
   
   cost = cost/(len(folds)-1)
   print("Cost={}".format(cost))

   validation_dataset = dataset[folds[validation_fold],:]
   accuracy = validate_model(nn, validation_dataset)

   return cost, accuracy


def validate_model(nn, dataset):
   accuracy = 0
   validation_size = dataset.shape[0]
   input_size = nn.layer_size[0]

   for i in range(validation_size):
      validation_example = dataset[i, :input_size]
      y_example = dataset[i, input_size:]

      out_full, out_layer = nn.classify(validation_example)
      
      if out_layer is None:
         print("Error in the NN. Output Layer == None")
         return 0.0

      y = np.around(out_layer)
      # print("ŷ={}, y={}".format(y.flatten(),  y_example.flatten()))
      if np.array_equal(y.flatten(), y_example.flatten()):
         accuracy += 1

   print("Acc: {} / {}".format(accuracy, validation_size))

   accuracy = accuracy/validation_size
   return accuracy

def plot_accuracy(accuracy, epoch, labels):
   pass

def plot_cost_function(error, epoch, labels):
   pass

