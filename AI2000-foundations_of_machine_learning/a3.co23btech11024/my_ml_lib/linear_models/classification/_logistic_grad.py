import numpy as np

class LogisticRegression_Grad_Desc:

    def __init__(self, eta=0.1, alpha=0.0, max_iter=100, tol=1e-5, fit_intercept=True):
        self.eta = eta
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w_ = None
        self.neg_ll = []

    def _sigmoid(self, w, x):
        """Numerically stable sigmoid function."""
        # Clip z to avoid overflow/underflow in exp
        z=x@w
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _neg_log_likelihood(self, X_aug, y, h):
        # This function calculates regularized negative log likelihood
        n_samples = y.shape[0]
        
        # Adding 1e-10 to avoid log(0)
        log_loss = -np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
        
        # L2 Regularization (not penalizing the intercept (w_[0])
        w_penalty = self.w_.copy()
        if self.fit_intercept:
            w_penalty[0, 0] = 0
        
        l2_penalty = (self.alpha / 2) * np.sum(w_penalty**2)
        cost = (log_loss + l2_penalty) / n_samples
        
        return cost

    
    def fit(self, X, y):
        # Fitting model using batch gradient descent.

            # Args:
            # X (np.ndarray): Training data, (n_samples, n_features)
            # y (np.ndarray): Target values (0 or 1), (n_samples,)

        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        self.neg_ll = []

        # Adding intercept term (if needed)
        if self.fit_intercept:
            b = np.ones((X.shape[0], 1))
            X_aug = np.hstack((b, X))
            self.w_ = np.zeros((X_aug.shape[1], 1))
        else:
            X_aug = X
            self.w_ = np.zeros((X_aug.shape[1], 1))
            
        # Gradient Descent 
        check = False
        for i in range(self.max_iter):
            w_old = self.w_.copy()

            # h = p(y=1|X), (n_samples, 1)
            h = self._sigmoid(self.w_, X_aug)
            gradient = (X_aug.T @ (h - y))/n_samples
            reg_term = self.alpha * self.w_
            
            if self.fit_intercept:
                reg_term[0, 0] = 0
            
            total_gradient = (gradient + reg_term) / n_samples
            
            # Updating weights using averaged gradient as log likelihood is averaged
            self.w_ = self.w_ - self.eta * total_gradient

            cost = self._neg_log_likelihood(X_aug, y, h)
            self.neg_ll.append(cost)

            weight_change = np.linalg.norm(self.w_ - w_old)
            if weight_change < self.tol:
                print(f"Converged after {i+1} iterations.")
                check = True
                break
        
        if not check:
             print(f"Warning: Gradient Descent did not converge within {self.max_iter} iterations.")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities [P(y=0|X), P(y=1|X)], shape (n_samples, 2).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # --- TODO: Step 1 - Augment X if fitting intercept ---
        # If self.fit_intercept was True during fit, augment X here too.
        X_aug = X # Placeholder

        if self.fit_intercept:
            b=np.ones((X.shape[0], 1))
            X_aug=np.hstack((b, X))

        # --- TODO: Step 2 - Calculate P(y=1 | X) ---
        # Calculate linear combination: 
        # Calculate probability: 
        prob_y1 = self._sigmoid(self.w_, X_aug).reshape(-1,1)

        # --- TODO: Step 3 - Calculate P(y=0 | X) ---
       
        prob_y0 = 1-prob_y1
        
        # --- TODO: Step 4 - Stack probabilities ---
        # Return shape should be (n_samples, 2)
        return np.hstack([prob_y0, prob_y1])

    def predict(self, X):
        """
        Predict class labels (0 or 1) for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (0 or 1), shape (n_samples,).
        """
        # --- TODO: Step 1 - Get P(y=1 | X) ---
        # Call self.predict_proba(X) and select the column corresponding to class 1.
        probabilities_y1 = self.predict_proba(X)[:, 1]

        # --- TODO: Step 2 - Apply threshold ---
        # Return 1 if probability >= 0.5, else 0.
        probabilities_y1-=0.5
        probabilities_y1=(probabilities_y1>=0)
        # Hint: Use boolean comparison and .astype(int)
        return probabilities_y1.astype(int) # Placeholder

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        # This method should work once predict() is implemented correctly.
        return np.mean(self.predict(X) == y)