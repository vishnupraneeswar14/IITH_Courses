# create_best_model.py

import numpy as np

try:
    from my_ml_lib.nn.modules.base import Module
    from my_ml_lib.nn.modules.linear import Linear
    from my_ml_lib.nn.modules.activations import ReLU
    from my_ml_lib.nn.modules.containers import Sequential
except ImportError:
    print("Error: Could not import from my_ml_lib.nn.")
    print("Please ensure your library is in the correct path.")
    # Attempt to import from the notebook's path as a fallback
    try:
        from my_ml_lib.nn.modules.base import Module
        from my_ml_lib.nn.modules.linear import Linear
        from my_ml_lib.nn.modules.activations import ReLU
        from my_ml_lib.nn.modules.containers import Sequential
    except ImportError as e:
        print(f"Fallback import also failed: {e}")



def initialize_best_model():
    n_features = 784
    n_classes = 10
    hidden_dim1 = 256
    hidden_dim2 = 128
    
    model = Sequential(
        Linear(n_features, hidden_dim1),   
        ReLU(),                            
        Linear(hidden_dim1, hidden_dim2),  
        ReLU(),                            
        Linear(hidden_dim2, n_classes)     
    )
    
    return model

if __name__ == "__main__":
    
    best_model = initialize_best_model()
    print(best_model)
    
    for name, param in best_model._get_named_parameters():
        print(f"  {name}: shape {param.data.shape}")