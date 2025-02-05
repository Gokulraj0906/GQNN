class DataSplitter:
    def __init__(self, X, y, train_size: float = 0.8,
                 shuffle: bool = True, random_state: int = None):
        """
        Initialize the data splitter with parameters for splitting data into training and testing sets.

        Args:
            X (DataFrame): Features dataframe.
            y (DataFrame): Targets dataframe.
            train_size (float, optional): Proportion of total samples allocated to training. Defaults to 0.8.
            shuffle (bool, optional): Whether to shuffle the indices before splitting. Defaults to True.
            random_state (int, optional): Seed for shuffling, ensuring reproducibility. Defaults to None.
        """
        import sys
        if sys.platform.startswith("linux"):
            import fireducks.pandas as pd  
        else:
            import pandas as pd  
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, (pd.DataFrame, pd.Series)):
            y = pd.DataFrame(y, columns=['Target'])
        if not isinstance(train_size, float) or not (0 <= train_size <= 1):
            raise ValueError("train_size must be a float between 0 and 1.")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean.")

        self.X = X
        self.y = y
        self.train_size = train_size
        self.shuffle = shuffle
        self.random_state = random_state

        self.n_samples = len(self.X)
        if self.n_samples != len(self.y):
            raise ValueError("X and y must have the same number of samples.")

    def split(self):
        """
        Split the data into training and testing sets according to the given parameters.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) where each variable is a DataFrame.
        """
        import numpy as np
        indices = np.arange(self.n_samples)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)  # Set seed for reproducibility
            indices = np.random.permutation(indices)

        train_size_int = int(self.train_size * self.n_samples)
        train_indices = indices[:train_size_int]
        test_indices = indices[train_size_int:]

        X_train, X_test = self.X.iloc[train_indices], self.X.iloc[test_indices]
        y_train, y_test = self.y.iloc[train_indices], self.y.iloc[test_indices]

        return X_train, X_test, y_train, y_test

    def __repr__(self):
        return f"DataSplitter with {self.n_samples} samples, train_size={self.train_size}, shuffle={self.shuffle}"
