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
        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components
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
    

# Example dataset
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

