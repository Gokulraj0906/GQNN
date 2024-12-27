import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

class MLModelValidator:
    def __init__(self, X, y, problem_type='classification', test_size=0.2, random_state=42):
        """
        Initialize the validator with data and problem type.
        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Target labels.
            problem_type (str): 'classification' or 'regression'.
            test_size (float): Fraction of data to use as test set.
            random_state (int): Random state for reproducibility.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.problem_type = problem_type
        self.models = self._initialize_models()

        # Standardize the dataset if applicable
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def _initialize_models(self):
        """
        Initialize models based on the problem type.
        Returns:
            dict: A dictionary of initialized models.
        """
        if self.problem_type == 'classification':
            return {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC()
            }
        elif self.problem_type == 'regression':
            return {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'SVM': SVR()
            }
        else:
            raise ValueError("Unsupported problem type. Choose 'classification' or 'regression'.")

    def validate(self, scoring_metric='accuracy'):
        """
        Validate all models and output their performance.
        Args:
            scoring_metric (str): Scoring metric for cross-validation.
        """
        results = {}
        for name, model in self.models.items():
            print(f"Training and validating {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            # Compute the relevant metrics
            if self.problem_type == 'classification':
                score = accuracy_score(self.y_test, y_pred)
                print(classification_report(self.y_test, y_pred))
            else:  # Regression
                score = mean_squared_error(self.y_test, y_pred)
                print(f"MSE for {name}: {score:.4f}")

            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=scoring_metric)
            results[name] = {
                'Test Score': score,
                'Cross-Validation Mean Score': np.mean(cv_scores),
                'Cross-Validation Std Dev': np.std(cv_scores),
            }

        # Display results
        print("\nModel Validation Results:")
        for name, metrics in results.items():
            print(f"\n{name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    from sklearn.datasets import make_classification, make_regression

    # For classification
    X_class, y_class = make_classification(n_samples=1000, n_features=20, random_state=42)

    # For regression
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    print("Classification Validation:")
    validator_class = MLModelValidator(X_class, y_class, problem_type='classification')
    validator_class.validate()

    print("\nRegression Validation:")
    validator_reg = MLModelValidator(X_reg, y_reg, problem_type='regression')
    validator_reg.validate()
