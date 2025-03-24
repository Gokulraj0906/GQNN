import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from GQNN.models.classification_model import QuantumClassifier_EstimatorQNN_CPU

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert to binary classification (Setosa vs Non-Setosa)
y = (y == 0).astype(int)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_qubits = X.shape[1]  # Use number of features as qubits
print(f"Number of qubits: {num_qubits}")
qnn = QuantumClassifier_EstimatorQNN_CPU(num_qubits=num_qubits, batch_size=16, lr=0.1)

# Train the model
print("Training the Quantum Neural Network...")
qnn.fit(X_train, y_train, epochs=10)

# Evaluate the model
accuracy = qnn.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

print(qnn.print_quantum_circuit())

# Save the model
qnn.save_model()

# # # # Load the model
# # # model = qnn.load_model()
# # # model.print_quantum_circuit()


# # model = QuantumClassifier_EstimatorQNN_CPU.load_model()
# # import torch

# # predictions = model.predict(X_test)
# # predicted_classes = (predictions > 0.5).astype(int)
# # print("Class Predictions:", predicted_classes)
# # softmax = torch.nn.Softmax(dim=1)
# # predictions_tensor = torch.tensor(predictions)
# # probabilities = softmax(predictions_tensor)  # Convert raw scores into probabilities
# # predicted_classes = torch.argmax(probabilities, dim=1).numpy()
# # print("Class Predictions:", predicted_classes)
# # print("Probabilities:", probabilities.detach().numpy())


# from GQNN.models.classification_model import QuantumClassifier_EstimatorQNN_CPU

# model = QuantumClassifier_EstimatorQNN_CPU.load_model()
# print(model.predict(X_test))