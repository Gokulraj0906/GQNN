# GQNN: A Python Package for Quantum Neural Networks
[![Publish Python Package](https://github.com/Gokulraj0906/GQNN/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/Gokulraj0906/GQNN/actions/workflows/pypi-publish.yml)
![Status](https://img.shields.io/badge/GQNN-Under%20Development-orange?style=for-the-badge&logo=github)
[![PyPI Downloads](https://static.pepy.tech/badge/gqnn/week)](https://pepy.tech/projects/gqnn)
[![PyPI Downloads](https://static.pepy.tech/badge/gqnn/month)](https://pepy.tech/projects/gqnn)
[![PyPI Downloads](https://static.pepy.tech/badge/gqnn)](https://pepy.tech/projects/gqnn)
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Build](https://github.com/gokulraj0906/GQNN/actions/workflows/python-package.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/Gokulraj0906/GQNN)
![GitHub Repo stars](https://img.shields.io/github/stars/Gokulraj0906/GQNN?style=social)

GQNN is a pioneering Python library designed for research and experimentation with Quantum Neural Networks (QNNs). By integrating principles of quantum computing with classical neural network architectures, GQNN enables researchers to explore hybrid models that leverage the computational advantages of quantum systems. This library was developed by **GokulRaj S** as part of his research on Customized Quantum Neural Networks.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [Use Cases](#use-cases)
6. [Documentation](#documentation)
7. [Requirements](#requirements)
8. [Contribution](#contribution)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)
11. [Contact](#contact)

---

## Introduction

Quantum Neural Networks (QNNs) are an emerging field of study combining the principles of quantum mechanics with artificial intelligence. The **GQNN** package offers a platform to implement and study hybrid quantum-classical neural networks, aiming to bridge the gap between theoretical quantum algorithms and practical machine learning applications.

This package allows you to:

- Experiment with QNN architectures.
- Train models on classical or quantum data.
- Explore quantum-enhanced learning algorithms.
- Conduct research in Quantum Machine Learning.

---

## Features

- **Hybrid Neural Networks**: Combines classical and quantum layers seamlessly.
- **Custom Quantum Circuits**: Design and implement your own quantum gates and circuits.
- **Lightweight and Flexible**: Built with Python, NumPy, and scikit-learn for simplicity and extensibility.
- **Scalable**: Easily scale models for larger qubit configurations or datasets.
- **Research-Oriented**: Ideal for academic and experimental use in quantum machine learning.

---

## Installation

### Prerequisites
- Python 3.7 to 3.12 higher or lower version is not supported
- Ensure pip is updated: `pip install --upgrade pip`

### Installing GQNN
#### From PyPI
```bash
pip install GQNN
```

#### From Source
```bash
git clone https://github.com/gokulraj0906/GQNN.git
cd GQNN
pip install .
```

---

## Getting Started

### Basic Example

### Classification model

```python
import matplotlib
matplotlib.use("Agg")
matplotlib.use("TkAgg")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from GQNN.models.classification_model import (
    QuantumClassifier_EstimatorQNN_CPU,
    QuantumClassifier_SamplerQNN_CPU,
    VariationalQuantumClassifier_CPU
)

# Data prep
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Helper to run, evaluate, save, visualize
def run_model(model, name):
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train, verbose=True)
    acc = model.score(X_test, y_test)
    print(f"{name} Accuracy: {acc:.4f}")
    model.save_model(f"{name.lower()}.pkl")
    model.print_model(f"{name.lower()}_circuit.png")

# Run different models
run_model(
    QuantumClassifier_EstimatorQNN_CPU(num_qubits=2, batch_size=32, lr=0.001),
    "EstimatorQNN"
)

run_model(
    QuantumClassifier_SamplerQNN_CPU(num_inputs=2, output_shape=2, ansatz_reps=1, maxiter=50),
    "SamplerQNN"
)

run_model(
    VariationalQuantumClassifier_CPU(num_inputs=2, maxiter=30),
    "VariationalQNN"
)
```

### Regression Example

 ```python
import matplotlib
matplotlib.use("Agg")

from GQNN.models.regression_model import (
    QuantumRegressor_EstimatorQNN_CPU,
    QuantumRegressor_VQR_CPU
)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate regression data
X, y = make_regression(
    n_samples=150,
    n_features=3,
    n_informative=3,
    noise=3.0,
    random_state=42,
    bias=0.0
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Helper
def run_regressor(model, name):
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_scaled, y_train_scaled, verbose=True)
    r2_train = model.score(X_train_scaled, y_train_scaled)
    r2_test = model.score(X_test_scaled, y_test_scaled)
    print(f"{name} R² Train: {r2_train:.4f}, R² Test: {r2_test:.4f}")
    model.save_model(f"{name.lower()}.pkl")
    model.print_model(f"{name.lower()}_circuit.png")

# Run models
run_regressor(
    QuantumRegressor_EstimatorQNN_CPU(num_qubits=3, maxiter=100),
    "EstimatorQNN_Regressor"
)

run_regressor(
    QuantumRegressor_VQR_CPU(num_qubits=3, maxiter=100),
    "VariationalQNN_Regressor"
)
 ```

 ### QSVM Example (Classification + Regression)

 ```python
"""
Comprehensive QSVM Testing: Classification and Regression
"""

from GQNN.models.qsvm import QSVC_CPU, QSVR_CPU
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, r2_score,
    mean_squared_error, mean_absolute_error
)
import numpy as np

def run_qsvc():
    X, y = make_classification(
        n_samples=80, n_features=2, n_informative=2,
        n_redundant=0, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = QSVC_CPU(num_qubits=2, feature_map_reps=2)
    model.fit(X_train_scaled, y_train, verbose=True)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nQSVC Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    model.save_model("qsvc_model.pkl")
    model.print_model("qsvc_circuit.png")

def run_qsvr():
    X, y = make_regression(n_samples=80, n_features=2, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = QSVR_CPU(num_qubits=2, feature_map_reps=2, epsilon=0.1)
    model.fit(X_train_scaled, y_train, verbose=True)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse, mae = mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)
    print(f"\nQSVR R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    model.save_model("qsvr_model.pkl")
    model.print_model("qsvr_circuit.png")

if __name__ == "__main__":
    run_qsvc()
    run_qsvr()
 ```

### Advanced Usage
For more advanced configurations, such as custom quantum gates or layers, refer to the [Documentation](#documentation).

---

## Use Cases

GQNN can be used for:
1. **Research and Development**: Experiment with quantum-enhanced machine learning algorithms.
2. **Education**: Learn and teach quantum computing principles via QNNs.
3. **Prototyping**: Develop proof-of-concept models for quantum computing applications.
4. **Hybrid Systems**: Integrate classical and quantum systems for real-world data processing.

---

## Documentation

Comprehensive documentation is available to help you get started with GQNN, including tutorials, API references, and implementation guides.

- **Documentation**: [GQNN Documentation](https://www.gokulraj.tech/GQNN/docs)
- **Examples**: [Examples Folder](https://www.gokulraj.tech/GQNN/examples)

---

## Requirements

The following dependencies are required to use GQNN:

- Python >= 3.7
- NumPy
- Pandas
- scikit-learn
- Qiskit
- Qiskit-machine-learning
- Qiskit_ibm_runtime
- matplotlib
- ipython
- pylatexenc

### For Linux Users
```bash
pip install GQNN[linux]
```

Optional:
- Quantum simulation tools (e.g., Qiskit or Cirq) for advanced quantum operations.

Install required dependencies using:
```bash
pip install GQNN
```

---

## Contribution

We welcome contributions to make GQNN better! Here's how you can contribute:

1. **Fork the Repository**: Click the "Fork" button on the GitHub page.
2. **Clone Your Fork**:
    ```bash
    git clone https://github.com/gokulraj0906/GQNN.git
    ```
3. **Create a New Branch**:
    ```bash
    git checkout -b feature-name
    ```
4. **Make Your Changes**: Implement your feature or bug fix.
5. **Push Changes**:
    ```bash
    git push origin feature-name
    ```
6. **Submit a Pull Request**: Open a pull request with a detailed description of your changes.

---

## License

GQNN is licensed under the  GPL-3.0 License. See the [LICENSE](LICENSE) file for full details.

---

## Acknowledgements

- This package is a result of research work by **GokulRaj S**.
- Special thanks to the open-source community and the developers of foundational quantum computing tools.
- Inspired by emerging trends in Quantum Machine Learning.

---

## Contact

For queries, feedback, or collaboration opportunities, please reach out:

**Author**: GokulRaj S  
**Email**: gokulsenthil0906@gmail.com  
**GitHub**: [gokulraj0906](https://github.com/gokulraj0906)  
**LinkedIn**: [Gokul Raj](https://www.linkedin.com/in/gokulraj0906)

---

Happy Quantum Computing! 🚀