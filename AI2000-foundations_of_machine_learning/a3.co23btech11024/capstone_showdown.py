import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import pickle

try:
    # Autograd Engine (Problem 4)
    from my_ml_lib.nn.autograd import Value
    from my_ml_lib.nn.modules.base import Module
    from my_ml_lib.nn.modules.linear import Linear
    from my_ml_lib.nn.modules.activations import ReLU, Sigmoid
    from my_ml_lib.nn.modules.containers import Sequential
    from my_ml_lib.nn.optim import SGD
    from my_ml_lib.nn.losses import CrossEntropyLoss

    # Preprocessing (Problem 3)
    from my_ml_lib.preprocessing._data import StandardScaler
    from my_ml_lib.preprocessing._gaussian import GaussianBasisFeatures
    from my_ml_lib.preprocessing._polynomial import PolynomialFeatures

    # Model Selection & Data (Problem 1 & 3)
    from my_ml_lib.model_selection._split import train_test_val_split
    from my_ml_lib.datasets._loaders import load_fashion_mnist 

    from my_ml_lib.linear_models.classification._logistic_grad import LogisticRegression_Grad_Desc
    
    print("Successfully imported all my_ml_lib components.")

except ImportError as e:
    print(f"Error importing from Final_BoilerPlate.my_ml_lib: {e}")
    print("Please ensure 'Final_BoilerPlate' is in your Python path and all required modules are implemented.")
    print("Required: _loaders.py, _split.py, _data.py, _gaussian.py, _polynomial.py")


def get_batches(X, y, batch_size, shuffle=True):
    # Generator function which yield mini-batches of data.
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_ind in range(0, n_samples, batch_size):
        end_ind = min(start_ind + batch_size, n_samples)
        batch_indices = indices[start_ind:end_ind]
        yield X[batch_indices], y[batch_indices]

# evaluate_autograd_model

def accuracy_autograd(model, X_test, y_test, batch_size=512):
    # For evaluating accuracy of an autograd model on test set.
    n_samples = X_test.shape[0]
    n_correct = 0
    
    # Processing in batches to avoid memory issues with large test sets
    for X_batch, y_batch in get_batches(X_test, y_test, batch_size, shuffle=False):
        # Forward pass
        y_pred_logits = model(X_batch)
        
        # Getting predictions (indices of max logit)
        y_pred_indices = np.argmax(y_pred_logits.data, axis=1)
        
        # Comparing with true labels
        n_correct += np.sum(y_pred_indices == y_batch)
        
    accuracy = n_correct / n_samples
    return accuracy

# train_autograd_model

def train_autograd(model, loss_model, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size, model_name="Model"):
    
    # Generic training loop for an autograd model
    
    train_losses = []
    val_accuracies = []
    best_val_acc = -1.0
    best_model_state = None 
    
    n_train_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_losses = []
        
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size, shuffle=True):
            optimizer.zero_grad()
            y_pred_logits = model(X_batch)
            loss = loss_model(y_pred_logits, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.data.item())
            epoch_losses.append(loss.data.item())
            
        avg_epoch_loss = np.mean(epoch_losses)
        
        # Validation Phase
        val_acc = accuracy_autograd(model, X_val, y_val, batch_size)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch}", f"Train Loss: {avg_epoch_loss:.4f}", f"Val Acc: {val_acc*100}%", f"Time: {epoch_time}s")
        
        # Saving Best Model(Based on validation accuracy)
        if val_acc > best_val_acc:
            print(f" Got better validation accuracy than current: {val_acc*100} > {best_val_acc*100}")
            best_val_acc = val_acc
            # Coping the entire model parameters' data instead of model object
            best_model_state = model.state_dict()

    print(f"Best Validation Accuracy: {best_val_acc*100}%")
    
    # Loading best performing weights back into the model
    if best_model_state:        
        current_params_dict = dict(model._get_named_parameters())
        
        for name, param in current_params_dict.items():
            if name in best_model_state:
                if param.data.shape == best_model_state[name].shape:
                    param.data[:] = best_model_state[name]
                else:
                    print(f"Mismatch in shape for {name} while loading best state")
        
    return train_losses, val_accuracies, best_val_acc

def load_and_prep_data(normalize_bool, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    
    X, y = load_fashion_mnist(kind='train', normalize=normalize_bool)
    print(X.shape)
    print(y.shape)
    # try:
        # X, y = load_fashion_mnist(kind='train')
        # print(X.shape)
        # print(y.shape)
    # except Exception as e:
    #     print(f"Could not load data using Final_BoilerPlate.my_ml_lib._loaders: {e}")
    #     sys.exit(1)
        
    # X is normalized by the loader, y is already 1D
    X = X.astype(np.float64) 
    # Ensuring float64 for autograd
    y = y.astype(int)
    
    print(f"Original data shape: X={X.shape}, y={y.shape}")

    # Using train_test_val_split
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(
        X, y, train_size=train_frac, val_size=val_frac, test_size=test_frac, shuffle=True,random_state=78
    )
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes  : X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

class OvRClassifier:
    
    def __init__(self, estimator, **params):
        
            # Args
            # estimator: Binary classifier class (most useful for k seperate logisitc classifiers in OvR Logistic Regression).
            # **params: Hyperparameters to pass to each estimator instance.
            # ** allows input to dict of arbitary length
        
        self.estimator_class = estimator
        self.params = params
        self.classifiers_ = []
        self.classes_ = None

    def fit(self, X, y):
        
        # Fitting K binary classifiers, one for each class.
            # Args:
            # X (np.ndarray): Training data, (n_samples, n_features)
            # y (np.ndarray): Target labels (0 to K-1), (n_samples,)
        
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        
        for k in self.classes_:
            # Creating a new binary target y_k (1 for class k, else 0)
            y_k = (y == k)
            y_k = y_k.astype(int)
            
            # Creating a new instance of the estimator for one of k classfiers to be part of OvR
            clf = self.estimator_class(**self.params)
            clf.fit(X, y_k)
            self.classifiers_.append(clf)
        
        return self

    def predict_proba_ovr(self, X):
        # Collecting P(y=1|X) from each of the K classifiers
        
        # .predict_proba is from _logistic.py
        all_probs = [c.predict_proba(X)[:, 1] for c in self.classifiers_]
        
        # Stacking all probabilities together and returning it
        return np.column_stack(all_probs)

    def predict(self, X):
        # Predicting the class label (classifier with max probability)
        probs = self.predict_proba_ovr(X)
        max_score_indices = np.argmax(probs, axis=1)
        return self.classes_[max_score_indices]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

X_train, y_train, X_val, y_val, X_test, y_test = load_and_prep_data(False, train_frac=0.7, val_frac=0.15, test_frac=0.15)

input_dim = X_train.shape[1] 
n_classes = 10                

final_results = {}
all_train_losses = {}
os.makedirs('saved_models', exist_ok=True)

lr_softmax_raw = 0.05
epochs_softmax_raw = 100
batch_size_softmax_raw = 1024

model_softmax_raw = Sequential(Linear(input_dim, n_classes))
loss_softmax_raw = CrossEntropyLoss(reduction='mean')
optimizer_softmax_raw = SGD(model_softmax_raw.parameters(), lr=lr_softmax_raw)

losses_raw, _, best_val_raw = train_autograd(
    model_softmax_raw, loss_softmax_raw, optimizer_softmax_raw, X_train, y_train, X_val, y_val, 
    epochs_softmax_raw, batch_size_softmax_raw, model_name="Softmax (Raw)"
)

test_acc_raw = accuracy_autograd(model_softmax_raw, X_test, y_test)
print(f"Test Accuracy (Softmax Raw): {test_acc_raw*100}%")

final_results['Softmax (Raw)'] = test_acc_raw
all_train_losses['Softmax (Raw)'] = losses_raw
model_softmax_raw.save_state_dict('saved_models/softmax_raw.npz')

X_train, y_train, X_val, y_val, X_test, y_test = load_and_prep_data(True, train_frac=0.7, val_frac=0.15, test_frac=0.15)

n_centers_rbf = 200
sigma_rbf = 5.0
lr_softmax_rbf = 0.005
epochs_softmax_rbf = 150
batch_size_softmax_rbf = 256

# Preprocessing RBF
scaler_rbf = StandardScaler()
rbf_transform  = GaussianBasisFeatures(n_centers=n_centers_rbf, sigma=sigma_rbf, random_state=42)
subset_indices = np.random.choice(X_train.shape[0], min(10000, X_train.shape[0]), replace=False)
rbf_transform.fit(X_train[subset_indices])

X_train_rbf = rbf_transform.transform(X_train)
X_val_rbf = rbf_transform.transform(X_val)
X_test_rbf = rbf_transform.transform(X_test)

X_train_rbf = scaler_rbf.fit_transform(X_train_rbf)
X_val_rbf = scaler_rbf.transform(X_val_rbf)
X_test_rbf = scaler_rbf.transform(X_test_rbf)



model_softmax_rbf = Sequential(Linear(n_centers_rbf, n_classes))
loss_softmax_rbf = CrossEntropyLoss(reduction='mean')
optimizer_softmax_rbf = SGD(model_softmax_rbf.parameters(), lr=lr_softmax_rbf)

losses_rbf, _, best_val_rbf = train_autograd(
    model_softmax_rbf, loss_softmax_rbf, optimizer_softmax_rbf, X_train_rbf, y_train, X_val_rbf, y_val,
    epochs_softmax_rbf, batch_size_softmax_rbf,model_name="Softmax (RBF)"
)

test_acc_rbf = accuracy_autograd(model_softmax_rbf, X_test_rbf, y_test)
print(f"Test Accuracy (Softmax RBF): {test_acc_rbf*100}%")

final_results['Softmax (RBF)'] = test_acc_rbf
all_train_losses['Softmax (RBF)'] = losses_rbf
model_softmax_rbf.save_state_dict('saved_models/softmax_rbf.npz')

hidden_dim1 = 256
hidden_dim2 = 128
lr_mlp = 0.05
epochs_mlp = 130
batch_size_mlp = 128

model_mlp = Sequential(
    Linear(input_dim, hidden_dim1),
    ReLU(),
    Linear(hidden_dim1, hidden_dim2),
    ReLU(),
    Linear(hidden_dim2, n_classes)
)

print("MLP Architecture:", model_mlp, sep="\n")

loss_mlp = CrossEntropyLoss(reduction='mean')
optimizer_mlp = SGD(model_mlp.parameters(), lr=lr_mlp)

losses_mlp, _, best_val_mlp = train_autograd(
    model_mlp, loss_mlp, optimizer_mlp, X_train, y_train, X_val, y_val,
    epochs_mlp, batch_size_mlp, model_name="MLP"
)

test_acc_mlp = accuracy_autograd(model_mlp, X_test, y_test)
print(f"Test Accuracy (MLP): {test_acc_mlp*100}%")

final_results['MLP'] = test_acc_mlp
all_train_losses['MLP'] = losses_mlp

# Saving the Overall Best Autograd Model
autograd_val_scores = {
    'softmax_raw': (best_val_raw, model_softmax_raw),
    'softmax_rbf': (best_val_rbf, model_softmax_rbf),
    'mlp': (best_val_mlp, model_mlp)
}

best_autograd_model_name = max(autograd_val_scores, key=lambda k: autograd_val_scores[k][0])
best_autograd_model_obj = autograd_val_scores[best_autograd_model_name][1]

print(f"Best autograd model on validation set: {best_autograd_model_name}")
best_autograd_model_obj.save_state_dict('saved_models/best_autograd_model.npz')

# Standardizing data
scaler_logistic = StandardScaler()
X_train_std = scaler_logistic.fit_transform(X_train)
X_val_std = scaler_logistic.transform(X_val)
X_test_std = scaler_logistic.transform(X_test)

#Tuning hyperparameters (alpha, eta, epochs)
eta=0.05
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
best_alpha = None
best_val_acc_ovr = -1.0
best_ovr_model = None

for alpha in alphas:
    ovr_model = OvRClassifier(LogisticRegression_Grad_Desc, eta=eta, alpha=alpha, max_iter=1000, tol=1e-6)
    ovr_model.fit(X_train_std, y_train)
    val_acc = ovr_model.score(X_val_std, y_val)
    
    print(f"Validation Accuracy for alpha={alpha}: {val_acc*100}%")
    
    if val_acc > best_val_acc_ovr:
        best_val_acc_ovr = val_acc
        best_alpha = alpha
        best_ovr_model = ovr_model
        
print(f"Best alpha: {best_alpha} (Val Acc: {best_val_acc_ovr*100}%)")

test_acc_ovr = best_ovr_model.score(X_test_std, y_test)
print(f"Test Accuracy (OvR Logistic): {test_acc_ovr*100}%")

final_results['OvR Logistic'] = test_acc_ovr

# Saving best OvR model
with open('saved_models/best_ovr_logistic_model.pkl', 'wb') as f:
    pickle.dump(best_ovr_model, f)

sorted_results = sorted(final_results.items(), key=lambda item: item[1], reverse=True)

for model_name, acc in sorted_results:
    print(f"{model_name} | {acc*100}")

# Moving average is considered for plotting losses in higher epochs (by anticipating same behaviour)
def moving_average(a, n=100):
    if len(a) < n:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
plt.figure(figsize=(8, 6))

steps_per_epoch = {}
steps_per_epoch['Softmax (Raw)'] = len(X_train) // batch_size_softmax_raw
steps_per_epoch['Softmax (RBF)'] = len(X_train_rbf) // batch_size_softmax_rbf
steps_per_epoch['MLP'] = len(X_train) // batch_size_mlp

for model_name, losses in all_train_losses.items():       
    smoothing_window = max(10, steps_per_epoch.get(model_name, 100) // 5)
    losses_smooth = moving_average(losses, n=smoothing_window)
    x_axis = np.linspace(0, len(losses), len(losses_smooth))
    plt.plot(x_axis, losses_smooth, label=model_name)

plt.title('Training Loss Curves (Smoothed) for Autograd Models')
plt.xlabel('Training Steps (Batches)')
plt.ylabel('Smoothed Cross-Entropy Loss')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

if 'Softmax (Raw)' in all_train_losses and all_train_losses['Softmax (Raw)']:
    plt.ylim(bottom=0, top=min(max(all_train_losses['Softmax (Raw)']), 2.5))

plt.savefig('capstone_training_loss_curves.png')
plt.show()

all_trained_models = {
    'MLP': model_mlp, 
    'Softmax (Raw)': model_softmax_raw,
    'Softmax (RBF)': model_softmax_rbf,
    'OvR Logistic': best_ovr_model 
}

best_model_name, best_model_acc = sorted_results[0]
best_model_object = all_trained_models[best_model_name]

print(f"The best overall model is: '{best_model_name}' with {best_model_acc*100:.2f}% test accuracy.")

save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

if best_model_name == 'OvR Logistic':
    # Saving OvR model using pickle (.pkl)
    save_path = os.path.join(save_dir, 'best_model.pkl')
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(best_model_object, f)
    except Exception as e:
        print(f"ERROR saving .pkl file: {e}")
        
else:
    # Saving state_dict for neural network based model (.npz)
    save_path = os.path.join(save_dir, 'best_model.npz')
    try:
        best_model_object.save_state_dict(save_path)
    except Exception as e:
        print(f"ERROR saving .npz file: {e}")