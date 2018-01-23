import numpy as np 

def k_fold(k, dataset_size):
   if k > dataset_size:
      return []

   permutation = np.random.permutation(
                     np.arange(dataset_size))
   folds = [permutation[i:i+k] for i in range(0, dataset_size, k)]
   return folds

