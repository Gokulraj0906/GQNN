class QuantumClassifier_EstimatorQNN_CPU:
    def __init__(self, num_qubits: int, maxiter: int = 50, batch_size: int = 32, lr: float = 0.001):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.circuit.library import QNNCircuit
        from qiskit.primitives import StatevectorEstimator as Estimator
        from qiskit_machine_learning.connectors import TorchConnector

        self.batch_size = batch_size
        self.num_qubits = num_qubits
        self.lr = lr
        self.qc = QNNCircuit(num_qubits)
        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(circuit=self.qc, estimator=self.estimator)
        self.model = TorchConnector(self.estimator_qnn)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
        print(f"Model initialized with {self.num_qubits} qubits, batch size {self.batch_size}, and learning rate {self.lr}")

    def fit(self, X, y, epochs=20, patience=5):
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm
        import torch
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print("Training in progress...\n")
        best_loss = float('inf')
        wait = 0
        training_losses = []

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
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg") 
        plt.figure(figsize=(8, 6))
        plt.plot(training_losses, label="Training Loss", color='b')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig("Training_Loss_Graph.png",dpi=3000)

    def predict(self, X):
        from sklearn.preprocessing import StandardScaler
        import torch
        X = StandardScaler().fit_transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()

        with torch.no_grad():
            raw_predictions = self.model(X_tensor)

        probabilities = torch.sigmoid(raw_predictions)
        predicted_classes = (probabilities > 0.5).int().cpu().numpy()

        return predicted_classes, probabilities.cpu().numpy()

    def score(self, X, y):
        import torch
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        predictions, _ = self.predict(X)
        accuracy = (predictions == y_tensor.numpy()).mean()
        return accuracy

    def save_model(self, file_path="quantumclassifier_estimatorqnn.pth"):
        import torch
        torch.save({
            'num_qubits': self.num_qubits,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path="quantumclassifier_estimatorqnn.pth", lr=0.001):
        import torch
        checkpoint = torch.load(file_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        num_qubits = checkpoint['num_qubits']
        model_instance = cls(num_qubits, lr=lr)

        model_instance.model.load_state_dict(checkpoint['model_state_dict'])
        model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Model loaded from {file_path}")
        return model_instance

    def print_quantum_circuit(self, file_name="quantum_EstimatorQNN_circuit.png"):
        from qiskit import QuantumCircuit
        decomposed_circuit = self.qc.decompose()
        decomposed_circuit.draw('mpl', filename=file_name)
        print(f"Decomposed circuit saved as {file_name}")
        print(f"The Circuit Is :\n{self.qc}")

""""This code will runs on Local computer """
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
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from qiskit.circuit.library import RealAmplitudes
        from qiskit.primitives import Sampler
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit_machine_learning.optimizers import COBYLA

        from qiskit_machine_learning.circuit.library import QNNCircuit
        import warnings
        warnings.filterwarnings("ignore")
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
        print(f"Model initialized with {self.num_inputs} qubits, batch size {self.batch_size}, and learning rate {self.lr}")


    def _initialize_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        import torch.nn.init as nn_init
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
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm
        import torch
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
        import matplotlib.pylab as plt 
        import matplotlib
        matplotlib.use("Agg")
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
        from sklearn.preprocessing import StandardScaler
        import torch
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
        import torch
        y_tensor = torch.tensor(y, dtype=torch.long).view(-1, 1)
        predictions, _ = self.predict(X)
        accuracy = (predictions == y_tensor.numpy()).mean()
        return accuracy

    def save_model(self, file_path="quantumclassifier_samplerqnn.pth"):
        """Save model to file."""
        import torch
        torch.save({
            'num_inputs': self.num_inputs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path="quantumclassifier_samplerqnn.pth", lr=0.001):
        """Load model from file."""
        import torch
        checkpoint = torch.load(file_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        num_inputs = checkpoint['num_inputs']
        model_instance = cls(num_inputs, lr=lr)

        model_instance.model.load_state_dict(checkpoint['model_state_dict'])
        model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Model loaded from {file_path}")
        return model_instance

    def print_quantum_circuit(self, file_name="quantum_SamplerQNN_circuit.png"):
        """Save and display the quantum circuit."""
        decomposed_circuit = self.qnn_circuit.decompose()
        decomposed_circuit.draw('mpl', filename=file_name)
        print(f"Decomposed circuit saved to {file_name}")
        print(f"The Quantum Circuit without a Decomposssion is:\n{self.qnn_circuit}")
"""
Variational Quantum Classifier (VQC) Implementation
This implementation provides a complete VQC with multiple feature maps, ansatz options,
and comprehensive training/evaluation capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2, TwoLocal
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class VariationalQuantumClassifier_CPU:
    """
    A comprehensive Variational Quantum Classifier implementation.
    
    Features:
    - Multiple feature map options (ZZ, Pauli, Custom)
    - Multiple ansatz options (RealAmplitudes, EfficientSU2, TwoLocal)
    - Flexible training with early stopping
    - Comprehensive evaluation metrics
    - Model saving/loading
    - Visualization tools
    """
    
    def __init__(self, 
                 num_features=4, 
                 num_qubits=None,
                 feature_map='zz',
                 ansatz='real_amplitudes',
                 reps=2,
                 lr=0.01,
                 batch_size=32,
                 random_state=42):
        """
        Initialize the Variational Quantum Classifier.
        
        Parameters:
        -----------
        num_features : int
            Number of input features
        num_qubits : int
            Number of qubits (defaults to num_features)
        feature_map : str
            Type of feature map ('zz', 'pauli', 'custom')
        ansatz : str
            Type of ansatz ('real_amplitudes', 'efficient_su2', 'two_local')
        reps : int
            Number of repetitions in the ansatz
        lr : float
            Learning rate
        batch_size : int
            Batch size for training
        random_state : int
            Random seed for reproducibility
        """
        
        self.num_features = num_features
        self.num_qubits = num_qubits if num_qubits else num_features
        self.feature_map_type = feature_map
        self.ansatz_type = ansatz
        self.reps = reps
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_history = []
        self.validation_history = []
        
        # Build quantum circuit
        self._build_quantum_circuit()
        
        # Initialize model
        self._initialize_model()
        
        print(f"VQC initialized with {self.num_qubits} qubits")
        print(f"Feature map: {feature_map}, Ansatz: {ansatz}, Reps: {reps}")
    
    def _build_quantum_circuit(self):
        """Build the quantum circuit with feature map and ansatz."""
        
        # Create feature map
        if self.feature_map_type == 'zz':
            self.feature_map = ZZFeatureMap(self.num_qubits, reps=1)
        elif self.feature_map_type == 'pauli':
            from qiskit.circuit.library import PauliFeatureMap
            self.feature_map = PauliFeatureMap(self.num_qubits, reps=1, paulis=['Z', 'ZZ'])
        elif self.feature_map_type == 'custom':
            self.feature_map = self._create_custom_feature_map()
        else:
            raise ValueError(f"Unknown feature map type: {self.feature_map_type}")
        
        # Create ansatz
        if self.ansatz_type == 'real_amplitudes':
            self.ansatz = RealAmplitudes(self.num_qubits, reps=self.reps)
        elif self.ansatz_type == 'efficient_su2':
            self.ansatz = EfficientSU2(self.num_qubits, reps=self.reps)
        elif self.ansatz_type == 'two_local':
            self.ansatz = TwoLocal(self.num_qubits, 'ry', 'cz', reps=self.reps)
        else:
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")
        
        # Combine feature map and ansatz
        self.quantum_circuit = self.feature_map.compose(self.ansatz)
        
        print(f"Quantum circuit created with {self.quantum_circuit.num_parameters} parameters")
    
    def _create_custom_feature_map(self):
        """Create a custom feature map."""
        qc = QuantumCircuit(self.num_qubits)
        
        # Add parameterized gates
        from qiskit.circuit import Parameter
        params = [Parameter(f'x_{i}') for i in range(self.num_features)]
        
        for i in range(self.num_qubits):
            if i < len(params):
                qc.ry(params[i], i)
        
        # Add entangling gates
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc
    
    def _initialize_model(self):
        """Initialize the quantum neural network and PyTorch connector."""
        
        # Create estimator
        self.estimator = Estimator()
        
        # Create quantum neural network
        self.qnn = EstimatorQNN(
            circuit=self.quantum_circuit,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator
        )
        
        # Create PyTorch connector
        self.model = TorchConnector(self.qnn)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
        print(f"Model initialized on device: {self.device}")
    
    def fit(self, X, y, validation_split=0.2, epochs=100, patience=10, verbose=True):
        """
        Train the VQC model.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        validation_split : float
            Fraction of data to use for validation
        epochs : int
            Maximum number of training epochs
        patience : int
            Early stopping patience
        verbose : bool
            Whether to print training progress
        """
        
        # Prepare data
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(np.unique(y_encoded))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded, test_size=validation_split, 
            random_state=self.random_state, stratify=y_encoded
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
            
            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Handle output dimensions
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0) if outputs.size(0) == 1 else outputs.unsqueeze(1)
                
                if self.num_classes == 2:
                    # Binary classification
                    loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), batch_y.float())
                    predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                else:
                    # Multi-class classification
                    if outputs.size(1) == 1:
                        # Need to create logits for each class
                        outputs_expanded = torch.zeros(outputs.size(0), self.num_classes).to(self.device)
                        outputs_expanded[:, 1] = outputs.squeeze()
                        outputs = outputs_expanded
                    
                    loss = self.loss_fn(outputs, batch_y)
                    predicted = torch.argmax(outputs, dim=1)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)
                
                if verbose:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100*train_correct/train_total:.2f}%'
                    })
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(0) if outputs.size(0) == 1 else outputs.unsqueeze(1)
                    
                    if self.num_classes == 2:
                        loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), batch_y.float())
                        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                    else:
                        if outputs.size(1) == 1:
                            outputs_expanded = torch.zeros(outputs.size(0), self.num_classes).to(self.device)
                            outputs_expanded[:, 1] = outputs.squeeze()
                            outputs = outputs_expanded
                        
                        loss = self.loss_fn(outputs, batch_y)
                        predicted = torch.argmax(outputs, dim=1)
                    
                    val_loss += loss.item()
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Store history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'val_acc': val_acc
            })
            
            if verbose:
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions on new data."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0) if outputs.size(0) == 1 else outputs.unsqueeze(1)
            
            if self.num_classes == 2:
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = (probabilities > 0.5).long()
            else:
                if outputs.size(1) == 1:
                    outputs_expanded = torch.zeros(outputs.size(0), self.num_classes).to(self.device)
                    outputs_expanded[:, 1] = outputs.squeeze()
                    outputs = outputs_expanded
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
        
        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions.cpu().numpy())
        return predictions_decoded, probabilities.cpu().numpy()
    
    def evaluate(self, X, y):
        """Evaluate the model on test data."""
        predictions, probabilities = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        cm = confusion_matrix(y, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return accuracy, report, cm
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.training_history:
            print("No training history available.")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history]
        train_accs = [h['train_acc'] for h in self.training_history]
        val_accs = [h['val_acc'] for h in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, train_losses, label='Training Loss', marker='o')
        ax1.plot(epochs, val_losses, label='Validation Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, label='Training Accuracy', marker='o')
        ax2.plot(epochs, val_accs, label='Validation Accuracy', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model."""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'num_features': self.num_features,
            'num_qubits': self.num_qubits,
            'num_classes': self.num_classes,
            'feature_map_type': self.feature_map_type,
            'ansatz_type': self.ansatz_type,
            'reps': self.reps,
            'training_history': self.training_history
        }
        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        self.num_classes = checkpoint['num_classes']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Model loaded from {filepath}")
    
    def get_circuit_info(self):
        """Get information about the quantum circuit."""
        info = {
            'num_qubits': self.num_qubits,
            'num_parameters': self.quantum_circuit.num_parameters,
            'depth': self.quantum_circuit.depth(),
            'feature_map': self.feature_map_type,
            'ansatz': self.ansatz_type,
            'reps': self.reps
        }
        return info
    
    def visualize_circuit(self):
        """Visualize the quantum circuit."""
        print("Quantum Circuit:")
        print(self.quantum_circuit.draw())
        return self.quantum_circuit.draw()