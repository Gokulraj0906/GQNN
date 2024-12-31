import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from QNN.data.rfe import FeatureSelector


data_classification = load_breast_cancer()
X_classification = pd.DataFrame(data_classification.data, columns=data_classification.feature_names)
y_classification = pd.Series(data_classification.target)



classification_selector = FeatureSelector(
    estimator=RandomForestClassifier(random_state=42),
    task='classification',
    step=1,
    cv=5
)


classification_selector.fit(X_classification, y_classification)


selected_features_classification = classification_selector.get_selected_features()
print("Selected features for classification:")
print("slength of selected features",classification_selector.get_ranking())
print(selected_features_classification)



X_transformed_classification = classification_selector.transform(X_classification)
print("Transformed classification dataset shape:", X_transformed_classification.shape)


data_regression = load_diabetes()
X_regression = pd.DataFrame(data_regression.data, columns=data_regression.feature_names)
y_regression = pd.Series(data_regression.target)


regression_selector = FeatureSelector(
    estimator=RandomForestRegressor(random_state=42),
    task='regression',
    step=1,
    cv=5
)


regression_selector.fit(X_regression, y_regression)


selected_features_regression = regression_selector.get_selected_features()
print("Selected features for regression:")
print("slength of selected features",regression_selector.get_ranking())
print(selected_features_regression)


X_transformed_regression = regression_selector.transform(X_regression)
print("Transformed regression dataset shape:", X_transformed_regression.shape)
