import numpy as np
import time
import itertools
import pandas as pd

class LinearRegression:
    """
    A class to perform Linear Regression from scratch.
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        """
        Initialize the Linear Regression handler.
        
        Args:
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.history = {"train_loss": []}

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Linear Regression model using gradient descent.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray or pd.Series): Training labels.
        """
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        
        # Ensure inputs are valid
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()  # Convert to 1D if necessary
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must match.")
        
        # Initialize weights and bias
        n_features = X_train.shape[1]
        self.weights = np.zeros((n_features, 1))  # Shape: (n_features, 1)
        self.bias = 0.0

        # Training process
        loading_symbols = itertools.cycle(["|", "/", "-", "\\"])
        for epoch in range(self.epochs):
            print(f"Training: Epoch {epoch + 1}/{self.epochs} {next(loading_symbols)}", end="\r")

            # Forward pass
            predictions = self._forward(X_train).ravel()

            # Compute Mean Squared Error (MSE) loss
            loss = np.mean((predictions - y_train) ** 2)
            self.history["train_loss"].append(loss)

            # Compute gradients
            error = predictions - y_train
            error = np.array(error)  # Error vector
            gradients_w = 2 * np.dot(X_train.T, error.reshape(-1, 1)) / X_train.shape[0]
            gradients_b = 2 * np.mean(error)

            # Update weights and bias
            self.weights -= self.learning_rate * gradients_w
            self.bias -= self.learning_rate * gradients_b

            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

        print("\nTraining complete!")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features.
        
        Returns:
            np.ndarray: Predicted values.
        """
        return self._forward(X).ravel()

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions (forward pass).
        
        Args:
            X (np.ndarray): Input features.
        
        Returns:
            np.ndarray: Predicted values.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model weights and bias are not initialized. Please call train() first.")
        return np.dot(X, self.weights) + self.bias

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Compute the R^2 score of the model.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray or pd.Series): True labels for the test set.
        
        Returns:
            float: R^2 score, a measure of how well the predictions match the true values.
        """
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()
        
        predictions = self.predict(X_test)

        tss = np.sum((y_test - np.mean(y_test)) ** 2)
        rss = np.sum((y_test - predictions) ** 2)

        r2_score = 1 - (rss / tss)
        return r2_score