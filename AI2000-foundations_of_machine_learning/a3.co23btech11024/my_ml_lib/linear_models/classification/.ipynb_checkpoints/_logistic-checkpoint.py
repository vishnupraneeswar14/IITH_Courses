# my_ml_lib/linear_models/classification/_logistic.py
import numpy as np

class LogisticRegression:
    """
    L2-Regularized Logistic Regression classifier using IRLS (Newton-Raphson).
    """
    def __init__(self, alpha=0.0, max_iter=100, tol=1e-5, fit_intercept=True):
        """
        Args:
            alpha (float): L2 regularization strength.
            max_iter (int): Maximum number of iterations for IRLS.
            tol (float): Tolerance for stopping criterion (change in weights).
            fit_intercept (bool): Whether to add a bias term.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w_ = None # Learned weights (includes intercept if fit_intercept is True)

    def _sigmoid(self, w, x):
        """Numerically stable sigmoid function."""
        # Clip z to avoid overflow/underflow in exp
        z=w.T@x.T
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        """
        Fit the L2-regularized logistic regression model using IRLS.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values (0 or 1), shape (n_samples,).
        """

        y=y.reshape(-1,1)

        # print("hello:",X.shape)
        n_samples, n_features = X.shape
        w_old = None # Keep track of previous weights for convergence check

        # --- TODO: Step 1 - Add intercept term (bias) ---
        
        if self.fit_intercept:
            b=np.ones((X.shape[0], 1))
            X_aug=np.hstack((b, X))
            self.w_=np.zeros((X_aug.shape[1],1))
        else:
            X_aug=X
            self.w_=np.zeros((X_aug.shape[1],1))

        
        # --- TODO: Step 2 - Regularization setup for IRLS ---
        # Create the regularization matrix: alpha * Identity
        reg_matrix=self.alpha*np.eye(X_aug.shape[1])
        if self.fit_intercept:
            reg_matrix[0][0]=0
        
       
        # --- TODO: Step 3 - IRLS Iterations ---
        check=True
        for i in range(self.max_iter):
            w_old = self.w_.copy() # Store weights from previous iteration

            # --- TODO: Step 3a - Calculate predictions (h) ---
            # Calculate the linear combination: 
            # Calculate the predicted probabilities: 
            
            # h = np.zeros(n_samples) # Placeholder
            h=self._sigmoid(w_old, X_aug)
            r_diag=h*(1-h)
            
            # --- TODO: Step 3b - Calculate weight matrix R (diagonal) ---
            # Ensure diagonal elements are not too close to zero (e.g., np.maximum(r_diag, 1e-10))
            # R = np.diag(r_diag)
            R=np.diag(r_diag.squeeze(0))
            R=np.maximum(R, 1e-10)
            h=h.T
            
            # --- TODO: Step 3c - Calculate Hessian (H) ---
            # H = X_aug^T @ R @ X_aug + reg_matrix
            hessian=X_aug.T@R@X_aug
            hessian+=reg_matrix
            
            # --- TODO: Step 3d - Calculate gradient (∇L) ---
            gradient=X_aug.T@(h-y)
            dummy=self.alpha*self.w_.copy()
            
            if self.fit_intercept:
                dummy[0][0]=0

            gradient+=dummy

            # --- TODO: Step 3e - Update weights ---
            # Solve the linear system H @ delta_w = gradient for delta_w
            # Hint: Use np.linalg.solve(hessian, gradient). Handle potential errors (np.linalg.LinAlgError)
            #       by possibly using np.linalg.pinv(hessian) @ gradient as a fallback, but print a warning.
            # Update weights: self.w_ = w_old - delta_w
        
            try:
                delta_w = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                print("Switching to pseudo-inverse as Hessian is (near) singular")
                delta_w = np.linalg.pinv(hessian) @ gradient

            self.w_ = w_old - delta_w  

        
            # --- TODO: Step 3f - Check for convergence ---
            # Calculate the norm of the change in weights: weight_change = np.linalg.norm(self.w_ - w_old)
            
            weight_change = np.linalg.norm(self.w_ - w_old)
            if weight_change < self.tol:
                print(f"Converged after {i+1} iterations.") # Optional convergence message
                check=False
                break

        # Optional: Add a warning if the loop finished without converging
        if check: # Runs if loop completes without break
            print(f"Warning: IRLS did not converge within {self.max_iter} iterations.")

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