"""
Quantum Machine Learning Regression Models

This module provides corrected implementations of quantum regressors using Qiskit
and Qiskit Machine Learning. Each class includes proper error handling, model
persistence, training progress visualization, and detailed documentation.

Classes:
    - QuantumRegressor_EstimatorQNN_CPU: Quantum regressor using EstimatorQNN
    - QuantumRegressor_VQR_CPU: Variational quantum regressor using VQR
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class QuantumRegressor_EstimatorQNN_CPU:
    """
    A quantum machine learning regressor using EstimatorQNN for regression tasks.
    
    This regressor utilizes a quantum neural network (QNN) with quantum circuits
    and employs classical optimization algorithms to train the quantum model for
    regression problems.
    
    Attributes:
        num_qubits (int): Number of qubits in the quantum circuit
        maxiter (int): Maximum number of optimization iterations
        qc (QNNCircuit): Quantum circuit representing the neural network
        estimator (Estimator): Estimator for quantum state measurements
        estimator_qnn (EstimatorQNN): Quantum neural network instance
        optimizer (SPSA): Optimization algorithm for training
        regressor (NeuralNetworkRegressor): Main regressor object
        weights (np.ndarray): Trained model weights
        objective_func_vals (list): Training objective function values
        
    Methods:
        fit: Train the quantum regressor
        predict: Make predictions on new data
        score: Evaluate model performance (R² score)
        save_model: Save trained model to disk
        load_model: Load model from disk
        print_model: Display and save quantum circuit
    """
    
    def __init__(self, num_qubits: int, maxiter: int = 30):
        """
        Initialize the quantum regressor with EstimatorQNN.
        
        Args:
            num_qubits (int): Number of qubits in the quantum circuit
            maxiter (int, optional): Maximum optimization iterations. Defaults to 30.
            
        Raises:
            ImportError: If required Qiskit packages are not installed
            RuntimeError: If quantum circuit initialization fails
            ValueError: If input parameters are invalid
        """
        try:
            from qiskit_machine_learning.optimizers import SPSA
            from qiskit_machine_learning.utils import algorithm_globals
            from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
            from qiskit_machine_learning.neural_networks import EstimatorQNN
            from qiskit_machine_learning.circuit.library import QNNCircuit
            from qiskit.primitives import StatevectorEstimator as Estimator
        except ImportError as e:
            raise ImportError(f"Required package not found: {e}. Please install qiskit-machine-learning.")
        
        # Validate inputs
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive")
        
        try:
            self.num_qubits = num_qubits
            self.maxiter = maxiter
            self.weights = None
            self.objective_func_vals = []
            
            # Set random seed for reproducibility
            algorithm_globals.random_seed = 42
            
            # Initialize quantum components
            self.qc = QNNCircuit(num_qubits)
            self.estimator = Estimator()
            self.estimator_qnn = EstimatorQNN(circuit=self.qc, estimator=self.estimator)
            
            # Initialize optimizer and regressor
            self.optimizer = SPSA(maxiter=maxiter)
            self.regressor = NeuralNetworkRegressor(
                neural_network=self.estimator_qnn,
                loss="absolute_error",
                optimizer=self.optimizer,
                callback=self._callback_graph
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize quantum regressor: {e}")
    
    def _callback_graph(self, weights: np.ndarray, obj_func_eval: float):
        """
        Callback to update the objective function during training.
        
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
        Train the quantum regressor on the provided data.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,) or (n_samples, 1)
            verbose (bool, optional): Whether to display progress. Defaults to True.
            
        Returns:
            np.ndarray: Final trained weights
            
        Raises:
            ValueError: If input data has invalid shape or type
            RuntimeError: If training fails
        """
        try:
            # Validate input data
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if y.ndim > 2 or (y.ndim == 2 and y.shape[1] != 1):
                raise ValueError("y must be 1D array or 2D array with single column")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            # Reshape y if necessary
            y = np.array(y).flatten()
            
            if verbose:
                print(f"Training quantum regressor with {len(X)} samples...")
                print(f"Features: {X.shape[1]}, Qubits: {self.num_qubits}")
                
                # Create progress bar for training iterations
                self._progress_bar = tqdm(total=self.maxiter, desc="Training", 
                                        unit="iter", leave=True)
            
            # Reset objective function values
            self.objective_func_vals = []
            
            # Train the regressor
            self.regressor.fit(X, y)
            self.weights = self.regressor.weights
            
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
            np.ndarray: Predicted target values
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If prediction fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Model must be trained before making predictions")
            
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            
            predictions = self.regressor.predict(X)
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the performance of the regressor using R² score.
        
        Args:
            X (np.ndarray): Test features of shape (n_samples, n_features)
            y (np.ndarray): True target values of shape (n_samples,)
            
        Returns:
            float: R² (coefficient of determination) score
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If evaluation fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Model must be trained before evaluation")
            
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if y.ndim > 1:
                raise ValueError("y must be a 1D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            r2_score = self.regressor.score(X, y)
            return r2_score
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def save_model(self, file_path: str = "quantum_regressor_estimator.pkl"):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str, optional): Path to save the model. Defaults to "quantum_regressor_estimator.pkl".
            
        Raises:
            RuntimeError: If model saving fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Cannot save untrained model")
            
            model_data = {
                'num_qubits': self.num_qubits,
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
    def load_model(cls, file_path: str = "quantum_regressor_estimator.pkl"):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str, optional): Path to the saved model. Defaults to "quantum_regressor_estimator.pkl".
            
        Returns:
            QuantumRegressor_EstimatorQNN_CPU: Loaded model instance
            
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
                maxiter=model_data['maxiter']
            )
            
            # Load weights and training history
            model_instance.weights = model_data['weights']
            model_instance.objective_func_vals = model_data.get('objective_func_vals', [])
            
            # Set the weights in the regressor
            model_instance.regressor.weights = model_data['weights']
            
            print(f"Model loaded from {file_path}")
            return model_instance
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _plot_training_curve(self, save_path: str = "estimator_regressor_training_curve.png"):
        """Plot and save the training objective function curve."""
        try:
            if not self.objective_func_vals:
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.objective_func_vals, 'b-', linewidth=2, label="Objective Function")
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Objective Function Value", fontsize=12)
            plt.title("Quantum Regressor Training Progress (EstimatorQNN)", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training curve saved to {save_path}")
            
        except Exception as e:
            print(f"Failed to save training curve: {e}")
    
    def print_model(self, file_name: str = "quantum_regressor_circuit_estimator.png"):
        """
        Display and save the quantum circuit diagram with model information.
        
        Args:
            file_name (str, optional): Filename to save the circuit diagram. 
                                     Defaults to "quantum_regressor_circuit_estimator.png".
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            # Save circuit diagram
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
            circuit = self.qc.decompose()
            circuit.draw(output='mpl', ax=ax, style='iqp')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Circuit image saved as {file_name}")
            
            # Print model information
            print(f"\nQuantum Regressor Information (EstimatorQNN):")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Circuit depth: {self.qc.depth()}")
            print(f"Number of parameters: {self.qc.num_parameters}")
            print(f"Maximum iterations: {self.maxiter}")
            print(f"\nModel Weights: {self.weights}")
            print(f"\nCircuit:\n{self.qc}")
            
        except Exception as e:
            print(f"Error displaying quantum circuit: {e}")


class QuantumRegressor_VQR_CPU:
    """
    A variational quantum regressor using Qiskit's Variational Quantum Regressor (VQR).
    
    This class implements a quantum circuit-based regression model that utilizes
    a feature map and ansatz to approximate continuous functions. It uses variational
    optimization to train the quantum parameters for regression tasks.

    Attributes:
        num_qubits (int): Number of qubits in the quantum circuit
        maxiter (int): Maximum number of optimization iterations
        objective_func_vals (list): Training objective function values
        estimator (Estimator): Quantum estimator for circuit evaluation
        feature_map (QuantumCircuit): Feature map encoding classical data
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        optimizer (L_BFGS_B): Classical optimizer for training
        regressor (VQR): Variational Quantum Regressor instance
        weights (np.ndarray): Learned weights of the trained model
    """
    
    def __init__(self, num_qubits: int, maxiter: int = 5):
        """
        Initialize the Variational Quantum Regressor.
        
        Args:
            num_qubits (int): Number of qubits to use in the quantum circuit
            maxiter (int, optional): Maximum optimization iterations. Defaults to 5.
            
        Raises:
            ImportError: If required Qiskit packages are not installed
            RuntimeError: If quantum circuit initialization fails
            ValueError: If input parameters are invalid
        """
        try:
            from qiskit_machine_learning.optimizers import L_BFGS_B
            from qiskit_machine_learning.utils import algorithm_globals
            from qiskit_machine_learning.algorithms.regressors import VQR
            from qiskit.primitives import StatevectorEstimator as Estimator
            from qiskit import QuantumCircuit
            from qiskit.circuit import Parameter
        except ImportError as e:
            raise ImportError(f"Required package not found: {e}. Please install qiskit-machine-learning.")
        
        # Validate inputs
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if maxiter <= 0:
            raise ValueError("maxiter must be positive")
        
        try:
            # Set random seed for reproducibility
            algorithm_globals.random_seed = 42
            
            self.num_qubits = num_qubits
            self.maxiter = maxiter
            self.objective_func_vals = []
            self.weights = None
            
            # Initialize quantum components
            self.estimator = Estimator()
            self._initialize_circuit()
            
            # Initialize optimizer and regressor
            self.optimizer = L_BFGS_B(maxiter=maxiter)
            self.regressor = VQR(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer,
                callback=self._callback_graph,
                estimator=self.estimator,
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize variational quantum regressor: {e}")
    
    def _initialize_circuit(self):
        """
        Initialize the quantum circuit with a feature map and an ansatz.
        
        The feature map encodes classical data into quantum states using RY rotations,
        and the ansatz is used for variational parameter optimization.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        
        # Create feature map with parameterized RY gates
        param_x = Parameter("x")
        self.feature_map = QuantumCircuit(self.num_qubits, name="FeatureMap")
        self.feature_map.ry(param_x, range(self.num_qubits))

        # Create ansatz with variational parameters
        param_y = Parameter("y")
        self.ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")
        self.ansatz.ry(param_y, range(self.num_qubits))
    
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
        Train the variational quantum regressor on the provided dataset.
        
        Args:
            X (np.ndarray): Training input data of shape (n_samples, n_features)
            y (np.ndarray): Target output values of shape (n_samples,) or (n_samples, 1)
            verbose (bool, optional): Whether to display progress. Defaults to True.
            
        Returns:
            np.ndarray: Final trained weights
            
        Raises:
            ValueError: If input data has invalid shape or type
            RuntimeError: If training fails
        """
        try:
            # Validate input data
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if y.ndim > 2 or (y.ndim == 2 and y.shape[1] != 1):
                raise ValueError("y must be 1D array or 2D array with single column")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            # Reshape y if necessary
            y = np.array(y).flatten()
            
            if verbose:
                print(f"Training variational quantum regressor with {len(X)} samples...")
                print(f"Features: {X.shape[1]}, Qubits: {self.num_qubits}")
                
                # Create progress bar for training iterations
                self._progress_bar = tqdm(total=self.maxiter, desc="Training", 
                                        unit="iter", leave=True)
            
            # Reset objective function values
            self.objective_func_vals = []
            
            # Train the regressor
            self.regressor.fit(X, y)
            self.weights = self.regressor.weights
            
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
        Make predictions using the trained variational quantum regressor.
        
        Args:
            X (np.ndarray): Input data for prediction of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted output values
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If prediction fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Model must be trained before making predictions")
            
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            
            predictions = self.regressor.predict(X)
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model's performance using R² score.
        
        Args:
            X (np.ndarray): Test input data of shape (n_samples, n_features)
            y (np.ndarray): True output values of shape (n_samples,)
            
        Returns:
            float: R² (coefficient of determination) score
            
        Raises:
            ValueError: If input data has invalid shape
            RuntimeError: If evaluation fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Model must be trained before evaluation")
            
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if y.ndim > 1:
                raise ValueError("y must be a 1D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            r2_score = self.regressor.score(X, y)
            return r2_score
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def save_model(self, file_path: str = "variational_quantum_regressor.pkl"):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str, optional): Path to save the model. Defaults to "variational_quantum_regressor.pkl".
            
        Raises:
            RuntimeError: If model saving fails or model not trained
        """
        try:
            if self.weights is None:
                raise RuntimeError("Cannot save untrained model")
            
            model_data = {
                'num_qubits': self.num_qubits,
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
    def load_model(cls, file_path: str = "variational_quantum_regressor.pkl"):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str, optional): Path to the saved model. Defaults to "variational_quantum_regressor.pkl".
            
        Returns:
            QuantumRegressor_VQR_CPU: Loaded model instance
            
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
                maxiter=model_data['maxiter']
            )
            
            # Load weights and training history
            model_instance.weights = model_data['weights']
            model_instance.objective_func_vals = model_data.get('objective_func_vals', [])
            
            # Set the weights in the regressor
            model_instance.regressor.weights = model_data['weights']
            
            print(f"Model loaded from {file_path}")
            return model_instance
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _plot_training_curve(self, save_path: str = "vqr_training_curve.png"):
        """Plot and save the training objective function curve."""
        try:
            if not self.objective_func_vals:
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.objective_func_vals, 'b-', linewidth=2, label="Objective Function")
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Objective Function Value", fontsize=12)
            plt.title("Variational Quantum Regressor Training Progress", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training curve saved to {save_path}")
            
        except Exception as e:
            print(f"Failed to save training curve: {e}")

    def print_model(self, file_name: str = "variational_quantum_regressor_circuit.png"):
        """
        Display and save the quantum circuit used in the model.
        
        Args:
            file_name (str, optional): Filename to save the circuit diagram. 
                                     Defaults to "variational_quantum_regressor_circuit.png".
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
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
            print(f"\nVariational Quantum Regressor Information:")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Feature map depth: {self.feature_map.depth()}")
            print(f"Ansatz depth: {self.ansatz.depth()}")
            print(f"Total parameters: {self.ansatz.num_parameters}")
            print(f"Maximum iterations: {self.maxiter}")
            print(f"\nModel Weights: {self.weights}")
            print(f"\nFeature Map:\n{self.feature_map}")
            print(f"\nAnsatz:\n{self.ansatz}")
            
        except Exception as e:
            print(f"Error displaying quantum circuit: {e}")