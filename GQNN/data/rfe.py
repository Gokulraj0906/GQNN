from sklearn.feature_selection import RFECV

class FeatureSelector:
    def __init__(self, estimator, task='classification', step=1, cv=5, scoring=None):
        """
        Initialize the FeatureSelector class.

        :param estimator: The model to use for feature selection.
        :param task: Type of task ('classification' or 'regression').
        :param step: Number of features to remove at each iteration.
        :param cv: Number of cross-validation folds.
        :param scoring: Scoring metric for cross-validation. Defaults based on the task.
        """
        self.estimator = estimator
        self.task = task
        self.step = step
        self.cv = cv
        self.scoring = scoring or ('accuracy' if task == 'classification' else 'neg_mean_squared_error')
        self.selector = None
        self.selected_features = None

    def fit(self, X, y):
        """
        Fit the RFECV to the data and select features.

        :param X: Feature matrix.
        :param y: Target vector.
        """
        self.selector = RFECV(
            estimator=self.estimator,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring
        )
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.support_]

    def transform(self, X):
        """
        Transform the dataset to include only selected features.

        :param X: Feature matrix.
        :return: Transformed feature matrix.
        """
        if self.selector is None:
            raise ValueError("The selector has not been fitted yet. Call `fit` first.")
        return X[self.selected_features]

    def fit_transform(self, X, y):
        """
        Fit the RFECV to the data and transform the dataset.

        :param X: Feature matrix.
        :param y: Target vector.
        :return: Transformed feature matrix.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self):
        """
        Get the names of selected features.

        :return: List of selected feature names.
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Call `fit` first.")
        return self.selected_features.tolist()

    def get_ranking(self):
        """
        Get the ranking of features.

        :return: Array of feature rankings.
        """
        if self.selector is None:
            raise ValueError("The selector has not been fitted yet. Call `fit` first.")
        return self.selector.ranking_

