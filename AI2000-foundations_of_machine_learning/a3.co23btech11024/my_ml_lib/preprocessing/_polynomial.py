import numpy as np
# May need itertools.combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        # Add any other attributes needed to store fitted info if necessary
        self.n_input_features_ = None
    
    def fit(self, X, y=None):
        # Usually fit doesn't do much here unless you need input dimensions
        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X):
        if self.n_input_features_ is None:
            raise RuntimeError("Transformer is not fitted yet. Call fit() first.")
        if X.shape[1] != self.n_input_features_:
            raise ValueError(f"Input has {X.shape[1]} features, but model was "
                             f"fitted with {self.n_input_features_} features.")

        # This list will hold all the new feature columns
        features_list = []
        n_samples, n_features = X.shape
        if self.include_bias:
            # Adding the bias column (degree 0)
            features_list.append(np.ones((n_samples, 1), dtype=X.dtype))
        
        for d in range(1, self.degree + 1):
            # Getting all combinations of feature indices of length d (similar to scikit-learn's implementation)
            combos = itertools.combinations_with_replacement(range(n_features), d)
            for combo in combos:
                # combo is a tuple of indices. ex, (0, 0) or (0, 2)
                # X[:, combo] selects the columns for the interaction
                new_feature = np.prod(X[:, combo], axis=1)
                features_list.append(new_feature.reshape(-1, 1))

        # Concatenating all generated feature columns
        X_poly = np.hstack(features_list)
        
        return X_poly

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)