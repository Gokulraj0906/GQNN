import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Initialize the PCA object.

        Parameters:
        n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Parameters:
        X (numpy.ndarray): The data matrix of shape (n_samples, n_features).
        """
        
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean


        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):
        """
        Project the data onto the principal components.

        Parameters:
        X (numpy.ndarray): The data matrix of shape (n_samples, n_features).

        Returns:
        numpy.ndarray: The transformed data matrix of shape (n_samples, n_components).
        """
        if self.components is None:
            raise RuntimeError("The PCA model must be fitted before transforming data.")
        
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Fit the PCA model to the data and transform it.

        Parameters:
        X (numpy.ndarray): The data matrix of shape (n_samples, n_features).

        Returns:
        numpy.ndarray: The transformed data matrix of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self):
        """
        Get the ratio of explained variance for each principal component.

        Returns:
        numpy.ndarray: Explained variance ratios for each principal component.
        """
        if self.explained_variance is None:
            raise RuntimeError("The PCA model must be fitted before getting explained variance ratio.")
        
        total_variance = np.sum(self.explained_variance)
        return self.explained_variance / total_variance