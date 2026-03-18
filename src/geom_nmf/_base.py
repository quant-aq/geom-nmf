import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class GeoNMF(BaseEstimator, RegressorMixin):
    """Geospatial Non-negative Matrix Factorization regressor.

    Parameters
    ----------
    n_components : int, default=10
        Number of latent components (NMF rank).
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_components=10, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GeoNMF
            Fitted estimator.
        """
        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """Predict target values for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)

        raise NotImplementedError("predict() is not yet implemented.")
