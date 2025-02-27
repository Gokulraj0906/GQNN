# GQNN: A Python Package for Quantum Neural Networks

GQNN is a pioneering Python library designed for research and experimentation with Quantum Neural Networks (GQNNs). By integrating principles of quantum computing with classical neural network architectures, GQNN enables researchers to explore hybrid models that leverage the computational advantages of quantum systems. This library was developed by **Gokul Raj S** as part of his research on Customized Quantum Neural Networks.

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

Quantum Neural Networks (GQNNs) are an emerging field of study combining the principles of quantum mechanics with artificial intelligence. The **GQNN** package offers a platform to implement and study hybrid quantum-classical neural networks, aiming to bridge the gap between theoretical quantum algorithms and practical machine learning applications.

This package allows you to:

- Experiment with GQNN architectures.
- Train models on classical or quantum data.
- Explore quantum-enhanced learning algorithms.
- Conduct research in Quantum Machine Learning.

---

## Features

- **Hybrid Neural Networks**: Combines classical and quantum layers seamlessly.
- **Custom Quantum Circuits**: Design and implement your quantum gates and circuits.
- **Lightweight and Flexible**: Built with Python, NumPy, and scikit-learn for simplicity and extensibility.
- **Scalable**: Easily scale models for larger qubit configurations or datasets.
- **Research-Oriented**: Ideal for academic and experimental use in quantum machine learning.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Ensure pip is updated: `pip install --upgrade pip`

### Installing GQNN
#### From PyPI (when published)
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
```python
from GQNN.data.dataset import Data_Read
from GQNN.models.data_split import DataSplitter
from GQNN.models.Linear_model import QuantumClassifier_EstimatorQNN
import numpy as np 

# Path to the dataset
data_dir = 'Employee_Salary_Dataset.csv'

# Read and preprocess the dataset
df = Data_Read.Read_csv(data_dir)

print("Original DataFrame (after reading and cleaning):")
print(df.head())

# Apply one-hot encoding to string columns
df_with_encoded_columns = Data_Read.convert_strings_to_numeric()
print("\nDataFrame after One-Hot Encoding of string columns:")
print(df_with_encoded_columns.head())

# Scale the dataset using Min-Max Scaling
scaled_df = Data_Read.Scale_data(method='minmax')
print("\nScaled DataFrame (using Min-Max Scaling):")
print(scaled_df.head())

# Split the dataset into features and target
x = df_with_encoded_columns.drop('Gender_Male', axis=1)
y = df_with_encoded_columns['Gender_Male'].astype(int)

# Split the data into training and testing sets
split = DataSplitter(x, y, test_size=0.75, shuffle=True, random_state=43)
x_train, x_test, y_train, y_test = split.split()

# Convert data to NumPy arrays for processing
x_train = np.array(x_train)
y_train = np.array(y_train)

# Initialize and train the Quantum Neural Network model
model = QuantumClassifier_EstimatorQNN(num_qubits=4, maxiter=60, random_seed=143)
model.fit(x_train, y_train)

# Print the trained model's parameters
model.print_model()

# Evaluate the model and compute accuracy
score = model.score(x_test, y_test)
adjusted_score = 1 - score
print(f"Model accuracy (adjusted): {adjusted_score * 100:.2f}%")

```

### Advanced Usage
For more advanced configurations, such as custom quantum gates or layers, refer to the [Documentation](#documentation).

---

## Use Cases

GQNN can be used for:
1. **Research and Development**: Experiment with quantum-enhanced machine learning algorithms.
2. **Education**: Learn and teach quantum computing principles via GQNNs.
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
- Qiski_ibm_runtime
- matplotlib
- ipython
- pylatexenc

## For Linux User 
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

GQNN is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

---

## Acknowledgements

- This package is a result of research work by **GokulRaj.S**.
- Special thanks to the open-source community and the developers of foundational quantum computing tools.
- Inspired by emerging trends in Quantum Machine Learning.

---

## Contact

For queries, feedback, or collaboration opportunities, please reach out:

**Author**: GokulRaj.S  
**Email**: gokulsenthil0906@gmail.com  
**GitHub**: [gokulraj0906](https://github.com/gokulraj0906)  
**LinkedIn**: [Gokul Raj](https://www.linkedin.com/in/gokulraj0906)

---

Happy Quantum Computing! 🚀