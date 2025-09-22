"""
Quantum Machine Learning Classification Models

This module provides corrected implementations of quantum classifiers using Qiskit
and Qiskit Machine Learning. Each class includes proper error handling, model
persistence, training progress visualization, and detailed documentation.

Classes:
    - QuantumClassifier_EstimatorQNN_CPU: Quantum classifier using EstimatorQNN
    - QuantumClassifier_SamplerQNN_CPU: Quantum classifier using SamplerQNN  
    - VariationalQuantumClassifier_CPU: Variational quantum classifier using VQC
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class QuantumClassifier_EstimatorQNN_CPU:
    """
    A quantum neural network classifier using EstimatorQNN and PyTorch integration.
    
    This classifier combines quantum circuits with classical neural networks through
    Qiskit's TorchConnector, enabling gradient-based training on quantum hardware
    or simulators.
    
    Attributes:
        num_qubits (int): Number of qubits in the quantum circuit
        batch_size (int): Batch size for training
        model (TorchConnector): Quantum-classical hybrid model
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        loss_fn (torch.nn.Module): Loss function for training
        device (torch.device): Computation device (CPU/GPU)
        scaler (StandardScaler): Data preprocessing scaler
        training_history (dict): Training metrics history
        
    Methods:
        fit: Train the quantum classifier
        predict: Make predictions on new data
        score: Evaluate model accuracy
        save_model: Save trained model to disk
        load_model: Load model from disk
        plot_training_graph: Visualize training progress
        print_model: Display and save quantum circuit
    """
    
    def __init__(self, num_qubits: int, maxiter: int = 50, batch_size: int = 32, lr: float = 0.001):
        """
        Initialize the quantum classifier with EstimatorQNN.
        
        Args:
            num_qubits (int): Number of qubits in the quantum circuit
            maxiter (int, optional): Maximum iterations (not used in this implementation). Defaults to 50.
            batch_size (int, optional): Training batch size. Defaults to 32.
            lr (float, optional): Learning rate for optimizer. Defaults to 0.001.
            
        Raises:
            ImportError: If required Qiskit packages are not installed
            RuntimeError: If quantum circuit initialization fails
        """
        try:
            from qiskit_machine_learning.neural_networks import EstimatorQNN
            from qiskit_machine_learning.circuit.library import QNNCircuit
            from qiskit.primitives import StatevectorEstimator as Estimator
            from qiskit_machine_learning.connectors import TorchConnector
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(f"Required package not found: {e}. Please install qiskit-machine-learning and scikit-learn.")
        
        # Initialize model parameters
        self.num_qubits = num_qubits
        self.batch_size = batch_size
        self.lr = lr
        self.training_history = {'losses': [], 'epochs': []}
        
        try:
            # Initialize quantum components
            self.qc = QNNCircuit(num_qubits)
            self.estimator = Estimator()
            self.estimator_qnn = EstimatorQNN(circuit=self.qc, estimator=self.estimator)
            self.model = TorchConnector(self.estimator_qnn)
            
            # Initialize classical components
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.scaler = StandardScaler()
            
            # Set device and initialize model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Initialize model parameters
            self._initialize_parameters()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize quantum classifier: {e}")
    
    def _initialize_parameters(self):
        """Initialize model parameters using Xavier uniform initialization."""
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, patience: int = 5, verbose: bool = True):
        """
        Train the quantum classifier on the provided data.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training labels of shape (n_samples,)
            epochs (int, optional): Number of training epochs. Defaults to 20.
            patience (int, optional): Early stopping patience. Defaults to 5.
            verbose (bool, optional): Whether to display progress bars. Defaults to True.
            
        Returns:
            dict: Training history containing losses and epochs
            
        Raises:
            ValueError: If input data has invalid shape or type
            RuntimeError: If training fails
        """
        try:
            from torch.utils.data import DataLoader, TensorDataset
            
            # Validate input data
            if X.ndim != 2 or y.ndim != 1:
                raise ValueError("X must be 2D array and y must be 1D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            if len(np.unique(y)) != 2:
                raise ValueError("This is a binary classifier. y must contain exactly 2 unique classes.")
            
            # Preprocess data
            X_scaled = self.scaler.fit_transform(X)
            
            # Create data loader
            dataset = TensorDataset(
                torch.tensor(X_scaled, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).view(-1, 1)
            )
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            if verbose:
                print(f"Training quantum classifier with {len(X)} samples...")
                print(f"Device: {self.device}, Qubits: {self.num_qubits}")
            
            # Training loop
            self.model.train()
            best_loss = float('inf')
            wait = 0
            
            for epoch in range(epochs):
                epoch_loss = 0
                num_batches = len(dataloader)
                
                # Progress bar for batches
                if verbose:
                    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", 
                                      unit="batch", leave=False)
                else:
                    progress_bar = dataloader
                
                for batch_X, batch_y in progress_bar:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(batch_X)
                    loss = self.loss_fn(output, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if verbose and hasattr(progress_bar, 'set_postfix'):
                        progress_bar.set_postfix(loss=f"{loss.item():.6f}")
                
                avg_loss = epoch_loss / num_batches
                self.training_history['losses'].append(avg_loss)
                self.training_history['epochs'].append(epoch + 1)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if verbose:
                print("Training completed!")
                self.plot_training_graph()
            
            return self.training_history
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray, return_probabilities: bool = False):
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            return_probabilities (bool, optional): Return prediction probabilities. Defaults to False.
            
        Returns:
            tuple or np.ndarray: If return_probabilities=True, returns (predictions, probabilities).
                               Otherwise, returns only predictions.
                               
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If prediction fails or model not trained
        """
        try:
            if not hasattr(self.scaler, 'mean_'):
                raise RuntimeError("Model must be trained before making predictions")
            
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            
            # Preprocess data
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                raw_predictions = self.model(X_tensor)
                probabilities = torch.sigmoid(raw_predictions)
                predicted_classes = (probabilities > 0.5).int().cpu().numpy().flatten()
                probabilities = probabilities.cpu().numpy().flatten()
            
            if return_probabilities:
                return predicted_classes, probabilities
            return predicted_classes
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the accuracy of the classifier.
        
        Args:
            X (np.ndarray): Test features of shape (n_samples, n_features)
            y (np.ndarray): True labels of shape (n_samples,)
            
        Returns:
            float: Accuracy score between 0 and 1
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If evaluation fails
        """
        try:
            if X.ndim != 2 or y.ndim != 1:
                raise ValueError("X must be 2D array and y must be 1D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            predictions = self.predict(X)
            accuracy = np.mean(predictions == y)
            return accuracy
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def save_model(self, file_path: str = "quantum_classifier_estimator.pkl"):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str, optional): Path to save the model. Defaults to "quantum_classifier_estimator.pkl".
            
        Raises:
            RuntimeError: If model saving fails
        """
        try:
            model_data = {
                'num_qubits': self.num_qubits,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler': self.scaler,
                'training_history': self.training_history
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, file_path: str = "quantum_classifier_estimator.pkl"):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str, optional): Path to the saved model. Defaults to "quantum_classifier_estimator.pkl".
            
        Returns:
            QuantumClassifier_EstimatorQNN_CPU: Loaded model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance
            model_instance = cls(
                num_qubits=model_data['num_qubits'],
                batch_size=model_data['batch_size'],
                lr=model_data['lr']
            )
            
            # Load states
            model_instance.model.load_state_dict(model_data['model_state_dict'])
            model_instance.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            model_instance.scaler = model_data['scaler']
            model_instance.training_history = model_data.get('training_history', {'losses': [], 'epochs': []})
            
            print(f"Model loaded from {file_path}")
            return model_instance
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def plot_training_graph(self, save_path: str = "quantum_training_loss.png"):
        """
        Plot and save the training loss curve.
        
        Args:
            save_path (str, optional): Path to save the plot. Defaults to "quantum_training_loss.png".
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.training_history['epochs'], self.training_history['losses'], 
                    'b-', linewidth=2, label="Training Loss")
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title("Quantum Classifier Training Progress", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training graph saved to {save_path}")
            
        except Exception as e:
            print(f"Failed to save training graph: {e}")
    
    def print_model(self, file_path: str = "quantum_circuit_estimator.png"):
        """
        Print and save the quantum circuit diagram.
        
        Args:
            file_path (str, optional): Path to save the circuit diagram. Defaults to "quantum_circuit_estimator.png".
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
            self.qc.decompose().draw(output='mpl', ax=ax, style='iqp')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Quantum circuit saved to {file_path}")
            print(f"\nQuantum Circuit Structure:")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Circuit depth: {self.qc.depth()}")
            print(f"Number of parameters: {self.qc.num_parameters}")
            print(f"\nCircuit:\n{self.qc}")
            
        except Exception as e:
            print(f"Error displaying quantum circuit: {e}")


class QuantumClassifier_SamplerQNN_CPU:
    """
    A quantum neural network classifier using SamplerQNN with parity interpretation.
    
    This classifier uses quantum sampling to measure qubit states and interprets
    the results through parity functions, suitable for binary classification tasks.
    
    Attributes:
        num_inputs (int): Number of input features/qubits
        output_shape (int): Number of output classes
        ansatz_reps (int): Number of repetitions in the ansatz circuit
        maxiter (int): Maximum optimization iterations
        sampler (StatevectorSampler): Quantum sampler for measurements
        qnn_circuit (QNNCircuit): Quantum neural network circuit
        qnn (SamplerQNN): Quantum neural network instance
        classifier (NeuralNetworkClassifier): Main classifier object
        objective_func_vals (list): Training objective function values
        weights (np.ndarray): Trained model weights
        
    Methods:
        fit: Train the quantum classifier
        score: Evaluate model accuracy  
        predict: Make predictions on new data
        save_model: Save trained model to disk
        load_model: Load model from disk
        print_model: Display and save quantum circuit
        parity: Static method for parity interpretation
    """
    
    def __init__(self, num_inputs: int, output_shape: int = 2, ansatz_reps: int = 1, maxiter: int = 30):
        """
        Initialize the quantum classifier with SamplerQNN.
        
        Args:
            num_inputs (int): Number of input features (qubits)
            output_shape (int, optional): Number of output classes. Defaults to 2.
            ansatz_reps (int, optional): Number of ansatz repetitions. Defaults to 1.
            maxiter (int, optional): Maximum optimization iterations. Defaults to 30.
            
        Raises:
            ImportError: If required Qiskit packages are not installed
            RuntimeError: If quantum circuit initialization fails
            ValueError: If input parameters are invalid
        """
        try:
            from qiskit.circuit.library import RealAmplitudes
            from qiskit_machine_learning.optimizers import COBYLA
            from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
            from qiskit_machine_learning.neural_networks import SamplerQNN
            from qiskit_machine_learning.circuit.library import QNNCircuit
            from qiskit.primitives import StatevectorSampler
        except ImportError as e:
            raise ImportError(f"Required package not found: {e}. Please install qiskit-machine-learning.")
        
        # Validate inputs
        if num_inputs <= 0:
            raise ValueError("num_inputs must be positive")
        if output_shape <= 0:
            raise ValueError("output_shape must be positive")
        if ansatz_reps <= 0:
            raise ValueError("ansatz_reps must be positive")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive")
        
        try:
            self.num_inputs = num_inputs
            self.output_shape = output_shape
            self.ansatz_reps = ansatz_reps
            self.maxiter = maxiter
            self.sampler = StatevectorSampler()
            self.objective_func_vals = []
            self.weights = None
            
            # Initialize quantum circuit
            self.qnn_circuit = QNNCircuit(ansatz=RealAmplitudes(self.num_inputs, reps=self.ansatz_reps))
            
            # Initialize quantum neural network
            self.qnn = SamplerQNN(
                circuit=self.qnn_circuit,
                interpret=self.parity,
                output_shape=self.output_shape,
                sampler=self.sampler,
            )
            
            # Initialize classifier
            self.classifier = NeuralNetworkClassifier(
                neural_network=self.qnn,
                optimizer=COBYLA(maxiter=maxiter),
                callback=self._callback_graph
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize quantum classifier: {e}")
    
    @staticmethod
    def parity(x: int) -> int:
        """
        Interpret the binary parity of the input integer.
        
        This function counts the number of 1s in the binary representation
        of the input and returns the parity (even=0, odd=1).
        
        Args:
            x (int): Input integer to compute parity for
            
        Returns:
            int: Parity of the input (0 or 1)
            
        Example:
            >>> QuantumClassifier_SamplerQNN_CPU.parity(5)  # 5 = '101' in binary
            1  # odd number of 1s
            >>> QuantumClassifier_SamplerQNN_CPU.parity(6)  # 6 = '110' in binary  
            0  # even number of 1s
        """
        return "{:b}".format(x).count("1") % 2
    
    def _callback_graph(self, weights: np.ndarray, obj_func_eval: float):
        """
        Callback to update the objective function graph during training.
        
        Args:
            weights (np.ndarray): Current model weights during training
            obj_func_eval (float): Current objective function value
        """
        self.objective_func_vals.append(obj_func_eval)
        
        # Update progress bar if available
        if hasattr(self, '_progress_bar'):
            self._progress_bar.set_postfix(
                obj_func=f"{obj_func_eval:.6f}",
                iteration=len(self.objective_func_vals)
            )
            self._progress_bar.update(1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Train the quantum classifier on the provided data.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training labels of shape (n_samples,)
            verbose (bool, optional): Whether to display progress. Defaults to True.
            
        Returns:
            np.ndarray: Final trained weights
            
        Raises:
            ValueError: If input data has invalid shape or type
            RuntimeError: If training fails
        """
        try:
            # Validate input data
            if X.ndim != 2 or y.ndim != 1:
                raise ValueError("X must be 2D array and y must be 1D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            if X.shape[1] != self.num_inputs:
                raise ValueError(f"X must have {self.num_inputs} features")
            
            if verbose:
                print(f"Training quantum classifier with {len(X)} samples...")
                print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
                
                # Create progress bar for training iterations
                self._progress_bar = tqdm(total=self.maxiter, desc="Training", 
                                        unit="iter", leave=True)
            
            # Reset objective function values
            self.objective_func_vals = []
            
            # Train the classifier
            self.classifier.fit(X, y)
            self.weights = self.classifier.weights
            
            if verbose:
                self._progress_bar.close()
                print(f"Training completed! Final objective: {self.objective_func_vals[-1]:.6f}")
                self._plot_training_curve()
            
            return self.weights
            
        except Exception as e:
            if hasattr(self, '_progress_bar'):
                self._progress_bar.close()
            raise RuntimeError(f"Training failed: {e}")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the accuracy of the classifier.
        
        Args:
            X (np.ndarray): Test features of shape (n_samples, n_features)
            y (np.ndarray): True labels of shape (n_samples,)
            
        Returns:
            float: Accuracy score between 0 and 1
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If evaluation fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Model must be trained before evaluation")
            
            if X.ndim != 2 or y.ndim != 1:
                raise ValueError("X must be 2D array and y must be 1D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            if X.shape[1] != self.num_inputs:
                raise ValueError(f"X must have {self.num_inputs} features")
            
            accuracy = self.classifier.score(X, y)
            return accuracy
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If prediction fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Model must be trained before making predictions")
            
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if X.shape[1] != self.num_inputs:
                raise ValueError(f"X must have {self.num_inputs} features")
            
            predictions = self.classifier.predict(X)
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def save_model(self, file_path: str = "quantum_classifier_sampler.pkl"):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str, optional): Path to save the model. Defaults to "quantum_classifier_sampler.pkl".
            
        Raises:
            RuntimeError: If model saving fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Cannot save untrained model")
            
            model_data = {
                'num_inputs': self.num_inputs,
                'output_shape': self.output_shape,
                'ansatz_reps': self.ansatz_reps,
                'maxiter': self.maxiter,
                'weights': self.weights,
                'objective_func_vals': self.objective_func_vals
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, file_path: str = "quantum_classifier_sampler.pkl"):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str, optional): Path to the saved model. Defaults to "quantum_classifier_sampler.pkl".
            
        Returns:
            QuantumClassifier_SamplerQNN_CPU: Loaded model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance
            model_instance = cls(
                num_inputs=model_data['num_inputs'],
                output_shape=model_data['output_shape'],
                ansatz_reps=model_data['ansatz_reps'],
                maxiter=model_data['maxiter']
            )
            
            # Load weights and training history
            model_instance.weights = model_data['weights']
            model_instance.objective_func_vals = model_data.get('objective_func_vals', [])
            
            # Set the weights in the classifier
            model_instance.classifier.weights = model_data['weights']
            
            print(f"Model loaded from {file_path}")
            return model_instance
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _plot_training_curve(self, save_path: str = "sampler_training_curve.png"):
        """Plot and save the training objective function curve."""
        try:
            if not self.objective_func_vals:
                return
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.use("TkAgg")
            plt.figure(figsize=(10, 6))
            plt.plot(self.objective_func_vals, 'b-', linewidth=2, label="Objective Function")
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Objective Function Value", fontsize=12)
            plt.title("Quantum Classifier Training Progress (SamplerQNN)", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training curve saved to {save_path}")
            
        except Exception as e:
            print(f"Failed to save training curve: {e}")
    
    def print_model(self, file_name: str = "quantum_circuit_sampler.png"):
        """
        Display and save the quantum circuit diagram with model information.
        
        Args:
            file_name (str, optional): Filename to save the circuit diagram. Defaults to "quantum_circuit_sampler.png".
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.use("TkAgg")
            # Save circuit diagram
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
            circuit = self.qnn_circuit.decompose()
            circuit.draw(output='mpl', ax=ax, style='iqp')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Circuit image saved as {file_name}")
            
            # Print model information
            print(f"\nQuantum Circuit Information:")
            print(f"Number of qubits: {self.num_inputs}")
            print(f"Circuit depth: {self.qnn_circuit.depth()}")
            print(f"Number of parameters: {self.qnn_circuit.num_parameters}")
            print(f"Ansatz repetitions: {self.ansatz_reps}")
            print(f"\nModel Weights: {self.weights}")
            print(f"\nCircuit:\n{self.qnn_circuit}")
            
        except Exception as e:
            print(f"Error displaying quantum circuit: {e}")


class VariationalQuantumClassifier_CPU:
    """
    A Variational Quantum Classifier (VQC) implementation using Qiskit.
    
    This classifier uses a feature map to encode classical data into quantum states
    and a parameterized ansatz circuit for classification. It employs variational
    optimization to train the quantum parameters.
    
    Attributes:
        num_inputs (int): Number of input features/qubits
        maxiter (int): Maximum optimization iterations
        feature_map (QuantumCircuit): Feature map for data encoding
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        sampler (StatevectorSampler): Quantum sampler for measurements
        vqc (VQC): Variational quantum classifier instance
        objective_func_vals (list): Training objective function values
        weights (np.ndarray): Trained model weights
        
    Methods:
        fit: Train the variational quantum classifier
        predict: Make predictions on new data
        score: Evaluate model accuracy
        save_model: Save trained model to disk
        load_model: Load model from disk
        print_model: Display and save quantum circuit
    """
    
    def __init__(self, num_inputs: int = 2, maxiter: int = 30):
        """
        Initialize the Variational Quantum Classifier.
        
        Args:
            num_inputs (int, optional): Number of input features/qubits. Defaults to 2.
            maxiter (int, optional): Maximum optimization iterations. Defaults to 30.
            
        Raises:
            ImportError: If required Qiskit packages are not installed
            RuntimeError: If quantum circuit initialization fails
            ValueError: If input parameters are invalid
        """
        try:
            from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
            from qiskit_machine_learning.algorithms.classifiers import VQC
            from qiskit_machine_learning.optimizers import COBYLA
            from qiskit.primitives import StatevectorSampler
        except ImportError as e:
            raise ImportError(f"Required package not found: {e}. Please install qiskit-machine-learning.")
        
        # Validate inputs
        if num_inputs <= 0:
            raise ValueError("num_inputs must be positive")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive")
        
        try:
            self.num_inputs = num_inputs
            self.maxiter = maxiter
            self.objective_func_vals = []
            self.weights = None
            
            # Initialize quantum components
            self.feature_map = ZZFeatureMap(num_inputs)
            self.ansatz = RealAmplitudes(num_inputs, reps=1)
            self.sampler = StatevectorSampler()
            
            # Initialize VQC
            self.vqc = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                loss="cross_entropy",
                optimizer=COBYLA(maxiter=maxiter),
                callback=self._callback_graph,
                sampler=self.sampler,
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize variational quantum classifier: {e}")
    
    def _callback_graph(self, weights: np.ndarray, obj_func_eval: float):
        """
        Callback function to track objective function during training.
        
        Args:
            weights (np.ndarray): Current model weights during training
            obj_func_eval (float): Current objective function value
        """
        self.objective_func_vals.append(obj_func_eval)
        
        # Update progress bar if available
        if hasattr(self, '_progress_bar'):
            self._progress_bar.set_postfix(
                obj_func=f"{obj_func_eval:.6f}",
                iteration=len(self.objective_func_vals)
            )
            self._progress_bar.update(1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Train the variational quantum classifier on the provided data.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training labels of shape (n_samples,)
            verbose (bool, optional): Whether to display progress. Defaults to True.
            
        Returns:
            np.ndarray: Final trained weights
            
        Raises:
            ValueError: If input data has invalid shape or type
            RuntimeError: If training fails
        """
        try:
            # Validate input data
            if X.ndim != 2 or y.ndim != 1:
                raise ValueError("X must be 2D array and y must be 1D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            if X.shape[1] != self.num_inputs:
                raise ValueError(f"X must have {self.num_inputs} features")
            
            # Ensure y is numpy array and has correct format
            y = np.array(y)
            
            if verbose:
                print(f"Training variational quantum classifier with {len(X)} samples...")
                print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
                
                # Create progress bar for training iterations
                self._progress_bar = tqdm(total=self.maxiter, desc="Training", 
                                        unit="iter", leave=True)
            
            # Reset objective function values
            self.objective_func_vals = []
            
            # Train the VQC
            self.vqc.fit(X, y)
            self.weights = self.vqc.weights
            
            if verbose:
                self._progress_bar.close()
                print(f"Training completed! Final objective: {self.objective_func_vals[-1]:.6f}")
                self._plot_training_curve()
            
            return self.weights
            
        except Exception as e:
            if hasattr(self, '_progress_bar'):
                self._progress_bar.close()
            raise RuntimeError(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If prediction fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Model must be trained before making predictions")
            
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if X.shape[1] != self.num_inputs:
                raise ValueError(f"X must have {self.num_inputs} features")
            
            predictions = self.vqc.predict(X)
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the accuracy of the classifier.
        
        Args:
            X (np.ndarray): Test features of shape (n_samples, n_features)
            y (np.ndarray): True labels of shape (n_samples,)
            
        Returns:
            float: Accuracy score between 0 and 1
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If evaluation fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Model must be trained before evaluation")
            
            if X.ndim != 2 or y.ndim != 1:
                raise ValueError("X must be 2D array and y must be 1D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            if X.shape[1] != self.num_inputs:
                raise ValueError(f"X must have {self.num_inputs} features")
            
            accuracy = self.vqc.score(X, y)
            return accuracy
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def save_model(self, file_path: str = "variational_quantum_classifier.pkl"):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str, optional): Path to save the model. Defaults to "variational_quantum_classifier.pkl".
            
        Raises:
            RuntimeError: If model saving fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Cannot save untrained model")
            
            model_data = {
                'num_inputs': self.num_inputs,
                'maxiter': self.maxiter,
                'weights': self.weights,
                'objective_func_vals': self.objective_func_vals
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, file_path: str = "variational_quantum_classifier.pkl"):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str, optional): Path to the saved model. Defaults to "variational_quantum_classifier.pkl".
            
        Returns:
            VariationalQuantumClassifier_CPU: Loaded model instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance
            model_instance = cls(
                num_inputs=model_data['num_inputs'],
                maxiter=model_data['maxiter']
            )
            
            # Load weights and training history
            model_instance.weights = model_data['weights']
            model_instance.objective_func_vals = model_data.get('objective_func_vals', [])
            
            # Set the weights in the VQC
            model_instance.vqc.weights = model_data['weights']
            
            print(f"Model loaded from {file_path}")
            return model_instance
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _plot_training_curve(self, save_path: str = "vqc_training_curve.png"):
        """Plot and save the training objective function curve."""
        try:
            if not self.objective_func_vals:
                return
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.use("TkAgg")
            plt.figure(figsize=(10, 6))
            plt.plot(self.objective_func_vals, 'b-', linewidth=2, label="Objective Function")
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Objective Function Value", fontsize=12)
            plt.title("Variational Quantum Classifier Training Progress", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training curve saved to {save_path}")
            
        except Exception as e:
            print(f"Failed to save training curve: {e}")
    
    def print_model(self, file_name: str = "variational_quantum_circuit.png"):
        """
        Display and save the quantum circuit diagram with model information.
        
        Args:
            file_name (str, optional): Filename to save the circuit diagram. Defaults to "variational_quantum_circuit.png".
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.use("TkAgg")
            # Save circuit diagram (feature map + ansatz)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
            
            # Draw feature map
            self.feature_map.draw(output='mpl', ax=ax1, style='iqp')
            ax1.set_title("Feature Map", fontsize=14)
            
            # Draw ansatz
            self.ansatz.draw(output='mpl', ax=ax2, style='iqp')
            ax2.set_title("Ansatz", fontsize=14)
            
            plt.tight_layout()
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Circuit diagram saved as {file_name}")
            
            # Print model information
            print(f"\nVariational Quantum Classifier Information:")
            print(f"Number of qubits: {self.num_inputs}")
            print(f"Feature map depth: {self.feature_map.depth()}")
            print(f"Ansatz depth: {self.ansatz.depth()}")
            print(f"Total parameters: {self.ansatz.num_parameters}")
            print(f"\nModel Weights: {self.weights}")
            print(f"\nFeature Map:\n{self.feature_map}")
            print(f"\nAnsatz:\n{self.ansatz}")
            
        except Exception as e:
            print(f"Error displaying quantum circuit: {e}")