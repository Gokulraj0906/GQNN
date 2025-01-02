import numpy as np
import pandas as pd

class DataSplitter:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, train_size: float, shuffle=True, random_state=None):
        """
        Initializes the DataSplitter class.

        Parameters:
        - X: Feature DataFrame
        - y: Target Series
        - train_size: Proportion of the dataset to include in the train split (default is 0.8)
        - shuffle: Whether to shuffle the data before splitting (default is True)
        - random_state: Seed for random number generator (default is None)
        """
        self.X = X
        self.y = y
        self.train_size = train_size
        self.shuffle = shuffle
        self.random_state = random_state

        self.n_samples = len(X)

        if self.random_state is not None:
            np.random.seed(self.random_state)
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be a boolean value")
        if not (0 < self.train_size <= 1):
            raise ValueError("train_size must be a float between 0 and 1")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if self.X.shape[0] == 0:
            raise ValueError("X and y must have at least one sample")
        if self.X.shape[0] == 1:
            raise ValueError("X and y must have more than one sample")
        if self.X.shape[1] == 0:
            raise ValueError("X must have at least one feature")
        if self.y.nunique() == 1:
            raise ValueError("y must have more than one unique value")

    def split(self):
        """
        Splits the data into training and testing sets.

        Returns:
        - X_train: Training features
        - X_test: Testing features
        - y_train: Training target
        - y_test: Testing target
        """
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:int(self.train_size * self.n_samples)]
        test_indices = indices[int(self.train_size * self.n_samples):]

        X_train = self.X.iloc[train_indices]
        X_test = self.X.iloc[test_indices]
        y_train = self.y.iloc[train_indices]
        y_test = self.y.iloc[test_indices]

        return X_train, X_test, y_train, y_test
