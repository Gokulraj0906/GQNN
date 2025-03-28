import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.optimizers import COBYLA
import torch.nn.init as nn_init
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from qiskit_machine_learning.circuit.library import QNNCircuit
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qiskit")
class QuantumClassifier_SamplerQNN_CPU:
    def __init__(self, num_qubits: int, batch_size: int = 32, lr: float = 0.001,
                 output_shape: int = 2, ansatz_reps: int = 1, maxiter: int = 30):
        """
        Initialize the QuantumClassifier with customizable parameters.

        Args:
            num_qubits (int): Number of inputs for the feature map and ansatz.
            batch_size (int): Batch size for training.
            lr (float): Learning rate.
            output_shape (int): Number of output classes for the QNN.
            ansatz_reps (int): Number of repetitions for the ansatz circuit.
            maxiter (int): Maximum iterations for the optimizer.
        """
        self.batch_size = batch_size
        self.num_inputs = num_qubits
        self.output_shape = output_shape
        self.ansatz_reps = ansatz_reps
        self.lr = lr

        self.qnn_circuit = QNNCircuit(ansatz=RealAmplitudes(self.num_inputs, reps=self.ansatz_reps))

        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.qnn_circuit,
            input_params=self.qnn_circuit.parameters[:self.num_inputs],
            weight_params=self.qnn_circuit.parameters[self.num_inputs:],
            output_shape=self.output_shape,
            sampler=self.sampler,
        )

        self.classifier = NeuralNetworkClassifier(
            neural_network=self.qnn,
            optimizer=COBYLA(maxiter=maxiter),
            callback=self.plot_training_graph
        )

        self.model = TorchConnector(self.classifier.neural_network)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._initialize_weights()

        try:
            if self.num_inputs < 4:
                raise ValueError("The Current Model is Able to Trains Under 4 Qubits please Do the Rfe And Select The 4 required Columns .")
            if self.batch_size is None:
                raise ValueError("batch_size must be specified.")
            if self.lr is None:
                raise ValueError("learning_rate must be specified.")
        except ValueError as e:
            print(f"Error: {e}")
        print(f"Model initialized with {self.num_inputs} qubits, batch size {self.batch_size}, and learning rate {self.lr}")


    def _initialize_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        for param in self.model.parameters():
            if param.dim() > 1:
                nn_init.xavier_uniform_(param)

    def fit(self, X, y, epochs: int = 50, patience: int = 50):
        """
        Train the QuantumClassifier on the provided dataset.
        Args:
            X (array-like): Training data.
            y (array-like): Training labels.
            epochs (int): Number of training epochs.
            patience (int): Early stopping patience.
        """
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        wait = 0
        training_losses = []

        print("Training in progress...\n")
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)

            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.loss_fn(output, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.plot_training_graph(training_losses)

    def plot_training_graph(self, training_losses):
        """Plot training loss graph."""
        plt.figure(figsize=(8, 6))
        plt.plot(training_losses, label="Training Loss", color='b')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig("Training_Loss_Graph.png", dpi=300)

    def predict(self, X):
        """Predict labels for given input data."""
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()

        with torch.no_grad():
            raw_predictions = self.model(X_tensor)

        probabilities = torch.sigmoid(raw_predictions)
        predicted_classes = (probabilities > 0.5).int().cpu().numpy()

        return predicted_classes, probabilities.cpu().numpy()

    def score(self, X, y):
        """Calculate accuracy of the model."""
        y_tensor = torch.tensor(y, dtype=torch.long).view(-1, 1)
        predictions, _ = self.predict(X)
        accuracy = (predictions == y_tensor.numpy()).mean()
        return accuracy

    def save_model(self, file_path="quantum_model.pth"):
        """Save model to file."""
        torch.save({
            'num_inputs': self.num_inputs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path="quantum_model.pth", lr=0.001):
        """Load model from file."""
        checkpoint = torch.load(file_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        num_inputs = checkpoint['num_inputs']
        model_instance = cls(num_inputs, lr=lr)

        model_instance.model.load_state_dict(checkpoint['model_state_dict'])
        model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Model loaded from {file_path}")
        return model_instance

    def print_quantum_circuit(self, file_name="quantum_circuit.png"):
        """Save and display the quantum circuit."""
        decomposed_circuit = self.qnn_circuit.decompose()
        decomposed_circuit.draw('mpl', filename=file_name)
        print(f"Decomposed circuit saved to {file_name}")
        print(f"The Quantum Circuit without a Decomposssion is:\n{self.qnn_circuit}")

# Example script for using the model
if __name__ == "__main__":
    # Load Diabetes dataset from sklearn
    diabetes = load_diabetes()
    X = diabetes.data[:, :5]
    y = (diabetes.target > diabetes.target.mean()).astype(int)  # Convert to binary classification (high or low)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply PCA to reduce the dimensionality to 4
    pca = PCA(n_components=5)
    X_train_reduced = pca.fit_transform(X_train)

    # Instantiate and train the model
    model = QuantumClassifier_SamplerQNN_CPU(num_qubits=5)

    model.fit(X_train_reduced, y_train, epochs=10)

    # Evaluate model
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save the model
    model.save_model()


    loaded_model = QuantumClassifier_SamplerQNN_CPU.load_model()

    # Predict with the loaded model
    predictions, probabilities = loaded_model.predict(X_test)
    print("Predictions:", predictions)
    print("Probabilities:", probabilities)

    # If predictions are multiclass, get the index of the max probability for each sample
    predictions = predictions.argmax(axis=1)  # Convert (89, 16) to (89,)

    # Now predictions and y_test should both have the shape (89,)
    print("y_test shape:", y_test.shape)
    print("predictions shape:", predictions.shape)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    # Print the quantum circuit
    loaded_model.print_quantum_circuit()