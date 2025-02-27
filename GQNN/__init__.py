"""
GQNN Package
============

The GQNN (Generalized Quantum Neural Networks) Python package, developed by GokulRaj S, 
is designed to facilitate the creation of **hybrid machine learning models** that integrate 
**Quantum Computing and Classical Machine Learning**. 

This package provides tools for **data preprocessing, model training, feature selection, 
dimensionality reduction, and validation** across both **classical and quantum** paradigms. 
Users can seamlessly train **classification** and **regression** models, leverage **Quantum 
Neural Networks (QNNs)**, and apply feature selection techniques like **Recursive Feature 
Elimination (RFE)** and **Principal Component Analysis (PCA)**.

---

Package Metadata:
-----------------
- **Author:** GokulRaj S  
- **Version:** 1.0.0  
- **License:** MIT  
- **Maintainer:** GokulRaj S  
- **Email:** gokulsenthil0906@gmail.com  
- **Status:** Development  
- **Description:** A Python package for Quantum and Classical Machine Learning models.  
- **Keywords:** Quantum Neural Networks, Quantum Computing, Machine Learning, Deep Learning, Neural Networks.  

Project URLs:
-------------
- **Homepage:** [GQNN Homepage](https://www.gokulraj.tech/GQNN)  
- **GitHub Repository:** [GitHub](https://github.com/gokulraj0906/GQNN)  
- **Documentation:** [GQNN Docs](https://www.gokulraj.tech/GQNN/docs)  
- **Bug Reports:** [Report Issues](https://www.gokulraj.tech/GQNN/report)  
- **Funding:** [Support the Project](https://www.gokulraj.tech/GQNN/support)  
- **Tutorials:** [GQNN Tutorials](https://www.gokulraj.tech/GQNN/tutorials)  

---

Main Features:
--------------
✅ **Quantum Neural Networks (QNNs)** for hybrid quantum-classical learning.  
✅ **Classical Machine Learning models** for classification and regression tasks.  
✅ **Feature selection** using Recursive Feature Elimination (RFE).  
✅ **Dimensionality reduction** with Principal Component Analysis (PCA).  
✅ **Dataset loading & preprocessing** utilities.  
✅ **Model evaluation & validation** tools.  
✅ **Seamless model saving & loading** functionalities.  

---

Example Usage:
--------------
```python
# Import necessary modules
from GQNN.data import dataset
from GQNN.models import classification_model, regression_model, data_split
from GQNN.models import save_models
from GQNN.validation import validation
from GQNN.data import rfe, pca

# Load the dataset
data = dataset.Data_Read.Read_csv('path/to/dataset.csv')

# Split the dataset into training and testing sets
splitter = data_split.DataSplitter(data, test_size=0.25, shuffle=True, random_state=42)
x_train, x_test, y_train, y_test = splitter.split()

# Train a Classification Model
classifier = classification_model.ClassificationModel()
classifier.fit(x_train, y_train)

# Train a Regression Model
regressor = regression_model.RegressionModel()
regressor.fit(x_train, y_train)

# Save the trained model
save_models.save_model(classifier, 'path/to/save/classifier.pkl')

# Validate the trained model
validation.validate_model(classifier, x_test, y_test)

# Perform Recursive Feature Elimination (RFE) for feature selection
rfe_selector = rfe.FeatureSelector(estimator=classifier, task="classification", step=1, cv=5)
rfe_selector.fit(x_train, y_train)

# Perform Principal Component Analysis (PCA) for dimensionality reduction
pca_transformer = pca.PCA(n_components=5)
x_train_transformed = pca_transformer.fit_transform(x_train)


For more details, visit the official documentation https://www.gokulraj.tech/GQNN/docs """

from GQNN.data import dataset
from GQNN.models import classification_model
from GQNN.models import data_split
from GQNN.models import regression_model
from GQNN.validation import validation
from GQNN.data import rfe
from GQNN.data import pca

__all__ = ["dataset","classification_model","data_split","rfe","pca","validation","regression_model"]

__author__ = "GokulRaj S"
__version__ = "1.0.0"
__license__ = "MIT"
__maintainer__ = "GokulRaj S"
__email__ = "gokulsenthil0906@gmail.com"
__status__ = "Development"
__description__ = "A Python package for Quantum Neural Networks"
__keywords__ = "Quantum Neural Networks, Quantum Computing, Machine Learning, Neural Networks"
__url__ = "https://www.gokulraj.tech/GQNN"
__github_url__ = "https://github.com/gokulraj0906/gqnn"
__documentation_url__ = "https://www.gokulraj.tech/gqnn_docs"
__bug_report_url__ = "https://www.gokulraj.tech/GQNN/report"
__funding_url__ = "https://www.gokulraj.tech/GQNN/support"
__tutorial_url__ = "https://www.gokulraj.tech/GQNN/tutorials"
