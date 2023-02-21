import numpy as np
from typing import Callable, Union

class LinearRegression:
    def __init__(self, 
                 n_iterations: int = 1000, 
                 lr: float = 0.01, 
                 batch_size = 16,
                 log_freq = 5,
                 seed: int = None):
        self.n_iterations = n_iterations
        self.lr = lr
        self.seed = seed
        self.log_freq = log_freq
        self.batch_size = batch_size
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Random initialized parameters
        self.W = np.random.rand(n_features)
        self.b = np.random.rand(1)
        self.losses = []
        for i in range(1, self.n_iterations + 1):
            
            # TODO: Shuffle dataset after each iteration, remember to map X and y correctly
            X = None
            y = None
            
            dataset_len = len(X) // self.batch_size
            loss = 0
            
            for batch_idx in range(dataset_len):
                
                # TODO: Split dataset into batches
                batched_X = None
                batched_y = None
                
                # TODO: Calculate the output
                y_pred = np.dot(None, None) + None
                
                # TODO: Calculate loss
                loss += None
                
                # TODO: Calculate gradients
                dW = None
                db = None
                
                # TODO: Update weights and bias
                self.W -= None
                self.b -= None
            
            avg_loss = loss / len(dataset_len)
            if i % self.log_freq == 0:
                print('Iteration {}: {:.5f}'.format(i, avg_loss))

            self.losses.append(loss / dataset_len)
    
    def predict(self, X):
        return np.dot(X, self.W) + self.b
    
    def score(self, X, y):
        # TODO: Evaluate more using R-Squared metric
        y_pred = None
        r2_score = 1 - None
        return r2_score