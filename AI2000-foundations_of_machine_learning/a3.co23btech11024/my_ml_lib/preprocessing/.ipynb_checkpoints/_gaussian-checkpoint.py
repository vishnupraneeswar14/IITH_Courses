import numpy as np

class GaussianBasisFeatures:
    def __init__(self, n_centers=100, sigma=5.0, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers_ = None
        self.random_state = random_state # For reproducibility if using random sampling

    def fit(self, X, y=None):
        """
        Select n_centers random points from X as RBF centers.
        """
        # TODO: Determine the centers (mu_j).
        # Strategy: Randomly sample n_centers points from X
        # print("GaussianBasisFeatures: Fit method needs implementation.")
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(X.shape[0], self.n_centers, replace=False)
        self.centers_ = X[indices]
        return self

    def transform(self, X):
        """
        Transform input X into Gaussian RBF features.
        Output shape: (n_samples, n_centers)
        """

        if self.centers_ is None:
            raise RuntimeError("Transformer is not fitted yet.")
        # TODO: Apply the RBF formula: exp(-(||X - center||^2 / (2 * sigma^2)))
      
    
        # Compute squared Euclidean distance between each sample and each center
        # Using broadcasting: ||x - mu||^2 = (x - mu)^2 summed over features
        # X shape: (n_samples, n_features)
        # centers_ shape: (n_centers, n_features)

        # This method was inefficient for large datasets like fashion mnist
        # dum=(X[:, :]-self.centers_[:, None, :])**2
        # dum=dum.sum(axis=2).T
        # dum/=(2*(self.sigma)**2)
        # dum=np.exp(dum)

        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2(x . c), followed this for memory efficient implementation
        
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        C_sq = np.sum(self.centers_**2, axis=1, keepdims=True).T
        dot_prod = X @ self.centers_.T
        
        sq_dists = X_sq + C_sq - 2 * dot_prod
        
        # nullying -ve floating point errors (if any)
        sq_dists = np.maximum(0, sq_dists)
        return np.exp(-sq_dists / (2 * self.sigma**2))

    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

