# my_ml_lib/model_selection/_split.py
import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split X and y arrays into random train and test subsets.

    Args:
        X (array-like): Feature data, shape (n_samples, n_features).
        y (array-like): Target labels, shape (n_samples,).
        test_size (float or int): Proportion (0.0-1.0) or absolute number for the test split.
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int, optional): Seed for shuffling reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # Setting random_state (if not set) for reproducibility
    if random_state==None:
        random_state=574

    # --- TODO: Step 1 - Calculate n_test and n_train ---
    # Check if test_size is float or int and calculate n_test.
    # Handle edge cases (invalid values for test_size).
    
    if test_size > 1 or test_size < 0:
            raise ValueError("Invalid Test Size (Should be b/w 0,1)")
        
    n_test  = int(n_samples*test_size) 
    n_train = n_samples - n_test


    # --- TODO: Step 2 - Create and Shuffle Indices ---
    if shuffle:
        dummy = np.hstack([X, y])
        np.random.shuffle(dummy)
        X=dummy[:, :-1]
        y=dummy[:, -1]
        

    # --- TODO: Step 3 - Split Indices ---into train_indices and test_indices
    rng=np.random.RandomState(random_state)
    indices_test = rng.choice(X.shape[0], n_test, replace=False)
    indices_train= np.arange(0, X.shape[0])
    indices_train= np.delete(indices_train, indices_test)
   

    # --- TODO: Step 4 - Split Arrays ---
    # print("Inside split.py")
    X_train = X[indices_train]
    y_train = y[indices_train]
    X_test = X[indices_test]
    y_test = y[indices_test]

    # print(X_train.shape, X_test.shape)

    return X_train, X_test, y_train, y_test


def train_test_val_split(X, y,
                         train_size=0.7,
                         val_size=0.15,
                         test_size=0.15,
                         shuffle=True,
                         random_state=None):
    """
    Split X and y arrays into random train, validation, and test subsets.

    Args:
        X (array-like): Feature data, shape (n_samples, n_features).
        y (array-like): Target labels, shape (n_samples,).
        train_size (float): Proportion for the train split (0.0 to 1.0).
        val_size (float): Proportion for the validation split (0.0 to 1.0).
        test_size (float): Proportion for the test split (0.0 to 1.0). Must sum to 1.0.
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int, optional): Seed for shuffling reproducibility.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # Setting random_state (if not set) for reproducibility
    if random_state==None:
        random_state=574

    # --- TODO: Step 1 - Validate Proportions ---
    # Check if train_size, val_size, test_size are valid floats between 0 and 1.
    if (test_size > 1 or test_size < 0) or (train_size > 1 or train_size < 0) or (val_size > 1 or val_size < 0):
            raise ValueError("Invalid Proportion sizes (Should be b/w 0,1)")
        
    # Check if they sum (approximately, use np.isclose) to 1.0. Raise ValueError if not.
    # if (np.isclose(test_size+ val_size + train_size, 1, atol=1e-2)):
    if not np.isclose(test_size + val_size + train_size, 1.0):
        # Allowing sum upto to 2 places inaccuracy
        raise ValueError("Proportion sizes dont sum to 1")
        

    # --- TODO: Step 2 - Calculate Split Sizes ---
    n_train=int(train_size*n_samples)
    n_val=int(val_size*n_samples)
    n_test = n_samples - n_train - n_val

    
    # Check if any calculated size is 0 and raise ValueError if so.
    if n_train == 0:
        raise ValueError("Train Set has no entries")
    if n_test == 0:
        raise ValueError("Test Set has no entries")
    if n_val == 0:
        raise ValueError("Validation Set has no entries")

    # --- TODO: Step 3 - Create and Shuffle Indices ---
    if shuffle:
        dummy = np.hstack([X, y])
        np.random.shuffle(dummy)
        X=dummy[:, :-1]
        y=dummy[:, -1]
        

    # --- TODO: Step 4 - Split Indices ---
    rng=np.random.RandomState(random_state)
    indices_test = rng.choice(X.shape[0], n_test, replace=False)
    indices_train= np.arange(0, X.shape[0])
    indices_train= np.delete(indices_train, indices_test)
    indices_val  = rng.choice(indices_train.shape[0], n_val, replace=False)
    indices_train= np.delete(indices_train, indices_val)


    # --- TODO: Step 5 - Split Arrays ---
    X_train, X_val, X_test = X[indices_train], X[indices_val], X[indices_test]
    y_train, y_val, y_test = y[indices_train], y[indices_val], y[indices_test]


    return X_train, X_val, X_test, y_train, y_val, y_test