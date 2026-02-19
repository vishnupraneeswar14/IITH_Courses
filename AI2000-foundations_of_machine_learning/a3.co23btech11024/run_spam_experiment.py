# run_spam_experiment.py

import numpy as np
import os

# --- Boilerplate: Imports ---
# Import necessary modules from your library and standard libraries
try:
    from my_ml_lib.datasets import load_spambase, DatasetNotFoundError
    from my_ml_lib.preprocessing import StandardScaler
    from my_ml_lib.linear_models.classification import LogisticRegression
    from my_ml_lib.model_selection import KFold, train_test_split
except ImportError as e:
    print(f"Error importing library components: {e}")
    print("Please ensure your my_ml_lib structure and __init__.py files are correct.")
    exit()
# --- End Boilerplate ---

# --- Boilerplate: Configuration ---
# Define constants for the experiment
DATA_FOLDER = "data/" # Directory containing spambase.data
TEST_SIZE = 0.2       # Proportion of data to use for the final test set
RANDOM_STATE = 42     # Seed for random operations (train/test split, KFold shuffle)
N_SPLITS_CV = 5       # Number of folds for cross-validation
ALPHAS_TO_TEST = [0.01, 0.1, 1, 10, 100] # L2 regularization strengths to test
# --- End Boilerplate ---


# --- TODO: Step 1 - Load Data ---
# Use the `load_spambase` function to load the dataset (X, y).
# Include error handling (try-except) for DatasetNotFoundError.
# Print the shapes of X and y after loading.

try:
    X, y = load_spambase(data_folder="data", filename="spambase.data", download_url=None)
    # print(X.shape, y.shape)
except:
    print(f"Error importing spambase data")
    print("Please ensure if spambase.dat file is present data directory.")
    exit()

# --- TODO: Step 2 - Split Data into Train and Test ---
# Use the `train_test_split` function to split X and y into
# X_train, X_test, y_train, y_test.
# Use TEST_SIZE, shuffle=True, and RANDOM_STATE for reproducibility.
# Print the shapes of the resulting train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE)

    
# # --- Helper Function for Cross-Validation --- ## TODO 
# def find_best_alpha(X_train_cv, y_train_cv, alphas, n_splits, random_state):
#         """Performs K-Fold CV to find the best alpha for Logistic Regression."""

#         best_alpha_found = alphas[0] # Placeholder
#         return best_alpha_found

def find_best_alpha(X_train_cv, y_train_cv, alphas, n_splits, random_state):
    """Performs K-Fold CV to find the best alpha for Logistic Regression."""

    
    best_alpha_found = -1
    best_mean_val_error = np.inf

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for alpha in alphas:
        fold_val_errors = []
        
        for train_ind, val_ind in kf.split(X_train_cv):
            # print("inside for loop:", len(train_ind), len(val_ind))
            X_train_fold, X_val_fold = X_train_cv[train_ind], X_train_cv[val_ind]
            # print(val_ind,X_train_cv.shape, X_val_fold.shape)
            # print(X_train_fold.shape)
            y_train_fold, y_val_fold = y_train_cv[train_ind], y_train_cv[val_ind]
            # print(y_train_cv.shape, y_train_fold.shape, y_val_fold.shape)

            # print(X_train_fold.shape, y_train_fold.shape)
            model = LogisticRegression(alpha=alpha)
            model.fit(X_train_fold, y_train_fold)
            
            # Calculating validation accuracy
            fold_accuracy = model.score(X_val_fold, y_val_fold)
            fold_val_errors.append(1.0 - fold_accuracy)

        mean_val_error = np.mean(fold_val_errors)
        print(f"Alpha= {alpha}, Mean CV Error= {mean_val_error}")

        if mean_val_error < best_mean_val_error:
            best_mean_val_error = mean_val_error
            best_alpha_found = alpha

    print(f"Best alpha found= {best_alpha_found}, (Error= {best_mean_val_error})")
    return best_alpha_found




# --- TODO: Step 4 - Experiment with RAW Data ---
# Call `find_best_alpha` using the training data (X_train, y_train).
# Store the result in `best_alpha_raw`.

best_alpha_raw = find_best_alpha(X_train, y_train, alphas=ALPHAS_TO_TEST, n_splits=N_SPLITS_CV, random_state=RANDOM_STATE)

# --- TODO: Step 5 - Train and Evaluate Final RAW Model ---

model_raw = LogisticRegression(alpha=best_alpha_raw)
model_raw.fit(X_train, y_train)

train_error_raw = 1.0 - model_raw.score(X_train, y_train)
test_error_raw  = 1.0 - model_raw.score(X_test, y_test)

print(f"Raw Model Train Error = {train_error_raw}")
print(f"Raw Model Test Error =  {test_error_raw}")


# --- TODO: Step 6 - Experiment with STANDARDIZED Data ---
scaler = StandardScaler()
scaler.fit(X_train) 

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

best_alpha_std = find_best_alpha(X_train_std, y_train, alphas=ALPHAS_TO_TEST, n_splits=N_SPLITS_CV, random_state=RANDOM_STATE)

# --- TODO: Step 7 - Train and Evaluate Final STANDARDIZED Model ---
# Instantiate `LogisticRegression` with the `best_alpha_std`.

model_std = LogisticRegression(alpha=best_alpha_std)
model_std.fit(X_train_std, y_train)

train_error_std = 1.0 - model_std.score(X_train_std, y_train)
test_error_std  = 1.0 - model_std.score(X_test_std, y_test)

# --- Boilerplate: Report Results ---
print("\n--- Summary Results ---")
print(f"Preprocessing | Best Alpha | Train Error | Test Error")
print(f":------------|-----------:|------------:|-----------:")
print(f"Raw           | {best_alpha_raw:<10} | {train_error_raw:<11.4f} | {test_error_raw:<10.4f}")
print(f"Standardized  | {best_alpha_std:<10} | {train_error_std:<11.4f} | {test_error_std:<10.4f}")
print("\nNOTE: Ensure the results above reflect your actual computed values.")
