"""
GQNN Package

This module is the core of the GQNN (Quantum Neural Networks) Python package, 
developed by GokulRaj S. It is designed for creating hybrid models that combine 
Quantum Computing and Neural Networks. The package provides tools for building, 
training, and evaluating customized Quantum Neural Networks.

Package Metadata:
-----------------
Author: GokulRaj S
Version: 0.1.0
License: MIT
Maintainer: GokulRaj S
Email: gokulsenthil0906@gmail.com
Status: Development
Description: A Python package for Quantum Neural Networks
Keywords: Quantum Neural Networks, Quantum Computing, Machine Learning, Neural Networks

URLs:
-----
Homepage: https://www.gokulraj.tech/GQNN
GitHub Repository: https://github.com/gokulraj0906/GQNN
Documentation: https://www.gokulraj.tech/GQNN/docs
Bug Reports: https://www.gokulraj.tech/GQNN/report
Funding: https://www.gokulraj.tech/GQNN/support
Tutorials: https://www.gokulraj.tech/GQNN/tutorials

Example Usage:
--------------
>>> from GQNN.data import dataset
>>> from GQNN.models import Linear_model
>>> from GQNN.models import save_models
>>> from GQNN.models import data_split
>>> from GQNN.validation import validation
>>> from GQNN.data import rfe
>>> from GQNN.data import pca

>>> # Load the dataset
>>> data = dataset.Data_Read.Read_csv('path/to/dataset.csv')

>>> # Split the dataset
>>> x_train, x_test, y_train, y_test = data_split.DataSplitter(data, 0.75, True, 42).split()

>>> # Train a Linear Model
>>> model = Linear_model.LinearModel()
>>> model.fit(x_train, y_train)

>>> # Save the trained model
>>> save_models.save_model(model, 'path/to/save/model.pkl')

>>> # Validate the model
>>> validation.validate_model(model, x_test, y_test)

>>> # Perform Recursive Feature Elimination
>>> rfe.FeatureSelector(estimator, task, step, cv).fit(x_train, y_train)

>>> # Perform Principal Component Analysis
>>> pca.PCA(n_components).fit_transform(x_train)

For more information, please refer to the documentation at https://www.gokulraj.tech/GQNN/docs
"""


from GQNN.data import dataset
from GQNN.models import Linear_model
from GQNN.models import data_split
from GQNN.validation import validation
from GQNN.data import rfe
from GQNN.data import pca

__all__ = ["dataset","Linear_model","data_split","rfe","pca","validation"]

__author__ = "GokulRaj S"
__version__ = "0.1.0"
__license__ = "MIT"
__maintainer__ = "GokulRaj S"
__email__ = "gokulsenthil0906@gmail.com"
__status__ = "Development"
__description__ = "A Python package for Quantum Neural Networks"
__keywords__ = "Quantum Neural Networks, Quantum Computing, Machine Learning, Neural Networks"
__url__ = "https://www.gokulraj.tech/GQNN"
__github_url__ = "https://github.com/gokulraj0906/GQNN"
__documentation_url__ = "https://www.gokulraj.tech/GQNN/docs"
__bug_report_url__ = "https://www.gokulraj.tech/GQNN/report"
__funding_url__ = "https://www.gokulraj.tech/GQNN/support"
__tutorial_url__ = "https://www.gokulraj.tech/GQNN/tutorials"
