import torch
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from GQNN.models.data_split import DataSplitter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.connectors import TorchConnector

# 🚀 Load and Preprocess Dataset
data = load_diabetes()
x = np.array(data.data)
y = np.array(data.target)

# ✅ Normalize features & target
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = (y - y.mean()) / y.std()

# 🔥 Split data
split = DataSplitter(x, y, train_size=0.75, shuffle=True, random_state=43)
x_train, x_test, y_train, y_test = split.split()

# Convert to numpy arrays
x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

# ✅ Ensure num_qubits = feature size
num_qubits = x_train.shape[1] 

# 🚀 Quantum Classifier
class QuantumClassifier_EstimatorQNN_CPU:
    def __init__(self, num_qubits: int, maxiter: int = 10, lr=0.05):
        self.qc = QNNCircuit(num_qubits)
        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(circuit=self.qc, estimator=self.estimator)

        # Convert QNN to a Torch model
        self.model = TorchConnector(self.estimator_qnn)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y, epochs=10, batch_size=32):
        """
        🚀 Efficient mini-batch training with dynamic loss visualization
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_fn = torch.nn.MSELoss()
        loss_history = []

        plt.ion()  # Enable live updating
        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Over Time")
        line, = ax.plot([], [], 'b-o')

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()

            loss_history.append(loss.item())

            # Update graph
            line.set_xdata(range(len(loss_history)))
            line.set_ydata(loss_history)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        plt.ioff()
        fig.savefig("training_loss.png")
        plt.close(fig)
        print("✅ Training complete! Loss graph saved as 'training_loss.png'.")

    def print_quantum_circuit(self):
        """
        Prints and saves the quantum circuit used in training.
        """
        print(self.qc)
        circuit_drawer(self.qc.decompose(), output='mpl', filename="quantum_circuit.png")
        print("✅ Quantum circuit diagram saved as 'quantum_circuit.png'.")

    def predict(self, X):
        """
        Predict using the trained model.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.numpy()

    def score(self, X, y):
        """
        Evaluate the model accuracy.
        """
        predictions = self.predict(X)
        accuracy = (predictions.round() == y).mean()
        return accuracy

# 🚀 Train the Quantum Model
print("🔥 Training the model...")
model = QuantumClassifier_EstimatorQNN_CPU(num_qubits=num_qubits, maxiter=10, lr=0.05)
model.fit(x_train, y_train, epochs=10, batch_size=32)

# ✅ Print and Save Quantum Circuit
model.print_quantum_circuit()


model_score = model.score(x_test, y_test)
print(f"🎯 Model accuracy: {model_score * 100:.2f}%")