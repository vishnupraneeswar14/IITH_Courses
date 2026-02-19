# my_ml_lib/model_selection/_kfold.py

import numpy as np 
import random

class KFold:
    """
    K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets.
    Splits dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation set while the k - 1 remaining
    folds form the training set.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        """
        Initializes the KFold splitter.

        Args:
            n_splits (int): Number of folds. Must be at least 2.
            shuffle (bool): Whether to shuffle the data before splitting into batches.
                            Note that shuffling is done each time split() is called.
            random_state (int, optional): When shuffle is True, random_state affects the ordering
                                           of the indices. Pass an int for reproducible output.
        """
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError("n_splits must be an integer greater than 1.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean value.")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X (array-like): Data to split, shape (n_samples, n_features).
            y (array-like, optional): The target variable for supervised learning problems.
                                      Used for stratification in StratifiedKFold, but not here.
            groups (array-like, optional): Group labels for the samples used while splitting the dataset
                                            into train/test set. Not used in standard KFold.

        Yields:
            tuple: (train_indices, test_indices) arrays for each split.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # Setting self.random_state (if not set) for reproducibility
        if self.random_state==None:
            self.random_state=574
        
        # TODO: Implement shuffling logic if self.shuffle is True
        if self.shuffle:
            dummy = np.hstack([X])
            np.random.shuffle(dummy)
            X_mod=dummy[:, :-1]
            
        # TODO: Implement the logic to calculate fold sizes and yield indices
        
        # Determine fold sizes, distributing remainder samples across first folds 
        # Hint  : Can use np.full  and then redestribute accordingly

        size=(X_mod.shape[0]//self.n_splits)

        rng=np.random.RandomState(self.random_state)
        # Forming k folds with integer sizes
        indices = rng.choice(X_mod.shape[0], (self.n_splits, size), replace=False)
        dummy = np.arange(0, X_mod.shape[0])
        dummy = np.delete(dummy, indices).reshape(-1,1)
        # dummy contains indices which are not part of k folds
        
        dummy_ind=rng.choice(indices.shape[0], ((dummy).shape[0]), replace=False)
        
        mod     = indices[dummy_ind]
        indices = np.delete(indices, dummy_ind, axis=0).tolist()
        
        # Distributing leftover indices (from dummy to original indices array)
        ind_mod = np.append(mod, dummy, axis=1).tolist()
        for i in range(len(ind_mod)):
            indices.append(ind_mod[i])
        
        random.shuffle(indices)

        # yield to make it a generator
        print(len(indices))
        for i in range(len(indices)):
            # print("inside kfold: ",(len(indices[:i] + indices[i+1:])), len(indices[i]))
            # count=0
            # for j in (indices[:i] + indices[i+1:]):
            #     count+=len(j)
            # count+=len(indices[i])

            # print(count)

            dum=(indices[:i] + indices[i+1:])
            dum1=[i for j in dum for i in j]
            
            yield (dum1, indices[i])


             

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits
    

