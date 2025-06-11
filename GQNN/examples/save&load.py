import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import ZFeatureMap, EfficientSU2
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# Set seed
algorithm_globals.random_seed = 42
torch.manual_seed(42)

# Step 1: Load and preprocess data
data = load_breast_cancer()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data.data)

# PCA to reduce to 4 features for 4 qubits
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 2: Quantum layer with TorchConnector
class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qubits = 4

        # Define feature map and ansatz
        feature_map = ZFeatureMap(self.num_qubits)
        ansatz = EfficientSU2(self.num_qubits, reps=2, entanglement="full")

        # Define QNN
        sampler = Sampler()
        qnn = EstimatorQNN(feature_map=feature_map, ansatz=ansatz, input_params=feature_map.parameters, weight_params=ansatz.parameters)
        
        # Torch-compatible layer
        self.qnn = TorchConnector(qnn)
    
    def forward(self, x):
        return self.qnn(x)

# Instantiate model, loss, optimizer
model = QuantumClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Train
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Step 4: Test
model.eval()
with torch.no_grad():
    test_output = model(X_test)
    predictions = torch.sigmoid(test_output).round()
    accuracy = (predictions == y_test).float().mean()
    print(f"\n✅ Test Accuracy: {accuracy.item() * 100:.2f}%")

# Step 5: Save model
torch.save(model.state_dict(), "quantum_model.pt")
print("✅ Model saved as 'quantum_model.pt'")

# Step 6: To load the model
# Uncomment this if you want to load later
# loaded_model = QuantumClassifier()
# loaded_model.load_state_dict(torch.load("quantum_model.pt"))
# loaded_model.eval()
# print("✅ Model loaded.")
