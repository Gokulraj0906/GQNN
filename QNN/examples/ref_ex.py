import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from QNN.data.rfe import FeatureSelector
# Load dataset for classification (e.g., breast cancer dataset)
data_classification = load_breast_cancer()
X_classification = pd.DataFrame(data_classification.data, columns=data_classification.feature_names)
y_classification = pd.Series(data_classification.target)

# Instantiate the FeatureSelector for classification
classification_selector = FeatureSelector(
    estimator=RandomForestClassifier(random_state=42),
    task='classification',
    step=1,
    cv=5
)

# Fit the selector to the classification dataset
classification_selector.fit(X_classification, y_classification)

# Get the selected features for classification
selected_features_classification = classification_selector.get_selected_features()
print("Selected features for classification:")
print("slength of selected features",classification_selector.get_ranking())
print(selected_features_classification)

# Transform the classification dataset
X_transformed_classification = classification_selector.transform(X_classification)
print("Transformed classification dataset shape:", X_transformed_classification.shape)

# Load dataset for regression (e.g., diabetes dataset)
data_regression = load_diabetes()
X_regression = pd.DataFrame(data_regression.data, columns=data_regression.feature_names)
y_regression = pd.Series(data_regression.target)

# Instantiate the FeatureSelector for regression
regression_selector = FeatureSelector(
    estimator=RandomForestRegressor(random_state=42),
    task='regression',
    step=1,
    cv=5
)

# Fit the selector to the regression dataset
regression_selector.fit(X_regression, y_regression)

# Get the selected features for regression
selected_features_regression = regression_selector.get_selected_features()
print("Selected features for regression:")
print("slength of selected features",regression_selector.get_ranking())
print(selected_features_regression)

# Transform the regression dataset
X_transformed_regression = regression_selector.transform(X_regression)
print("Transformed regression dataset shape:", X_transformed_regression.shape)
