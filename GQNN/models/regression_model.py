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
    Enhanced quantum machine learning regressor using EstimatorQNN.
    """
    
    def __init__(self, num_qubits: int, maxiter: int = 30, learning_rate: float = 0.1):
        """
        Initialize the quantum regressor with EstimatorQNN.
        
        Args:
            num_qubits (int): Number of qubits in the quantum circuit
            maxiter (int, optional): Maximum optimization iterations. Defaults to 30.
            learning_rate (float, optional): Learning rate for SPSA optimizer. Defaults to 0.1.
        """
        try:
            from qiskit_machine_learning.optimizers import SPSA, COBYLA
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
            self.learning_rate = learning_rate
            self.weights = None
            self.objective_func_vals = []
            self._iteration_count = 0
            
            # Set random seed for reproducibility
            algorithm_globals.random_seed = 42
            
            # Initialize quantum components
            self.qc = QNNCircuit(num_qubits)
            self.estimator = Estimator()
            self.estimator_qnn = EstimatorQNN(circuit=self.qc, estimator=self.estimator)
            
            # Use COBYLA optimizer (more reliable callbacks than SPSA)
            self.optimizer = COBYLA(maxiter=maxiter)
            
            self.regressor = NeuralNetworkRegressor(
                neural_network=self.estimator_qnn,
                loss="squared_error",  # Changed to squared_error for better convergence
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
        self._iteration_count += 1
        self.objective_func_vals.append(obj_func_eval)
        
        # Update progress bar if available
        if hasattr(self, '_progress_bar') and self._progress_bar is not None:
            self._progress_bar.set_postfix(
                obj_func=f"{obj_func_eval:.6f}",
                iteration=self._iteration_count
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
                print(f"Optimizer: COBYLA, Max iterations: {self.maxiter}")
                print(f"Note: Training may take a few minutes...\n")
                
                # Create progress bar for training iterations
                self._progress_bar = tqdm(total=self.maxiter, desc="Training", 
                                        unit="iter", leave=True, 
                                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                self._progress_bar = None
            
            # Reset objective function values and iteration count
            self.objective_func_vals = []
            self._iteration_count = 0
            
            # Train the regressor
            self.regressor.fit(X, y)
            self.weights = self.regressor.weights
            
            if verbose:
                if self._progress_bar is not None:
                    # Update progress bar to show completion
                    remaining = self.maxiter - self._iteration_count
                    if remaining > 0:
                        self._progress_bar.update(remaining)
                    self._progress_bar.close()
                
                # Display results
                if self.objective_func_vals:
                    print(f"\n✓ Training completed!")
                    print(f"  Total iterations: {len(self.objective_func_vals)}")
                    print(f"  Final objective: {self.objective_func_vals[-1]:.6f}")
                    print(f"  Initial objective: {self.objective_func_vals[0]:.6f}")
                    print(f"  Improvement: {self.objective_func_vals[0] - self.objective_func_vals[-1]:.6f}")
                    self._plot_training_curve()
                else:
                    print(f"\n✓ Training completed!")
                    print(f"  (Callback data not available - this is normal for some optimizers)")
            
            return self.weights
            
        except Exception as e:
            if hasattr(self, '_progress_bar') and self._progress_bar is not None:
                self._progress_bar.close()
            raise RuntimeError(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted target values
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
            file_path (str, optional): Path to save the model.
        """
        try:
            if self.weights is None:
                raise RuntimeError("Cannot save untrained model")
            
            model_data = {
                'num_qubits': self.num_qubits,
                'maxiter': self.maxiter,
                'learning_rate': self.learning_rate,
                'weights': self.weights,
                'objective_func_vals': self.objective_func_vals
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✓ Model saved to {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, file_path: str = "quantum_regressor_estimator.pkl"):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str, optional): Path to the saved model.
            
        Returns:
            QuantumRegressor_EstimatorQNN_CPU: Loaded model instance
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance
            model_instance = cls(
                num_qubits=model_data['num_qubits'],
                maxiter=model_data['maxiter'],
                learning_rate=model_data.get('learning_rate', 0.1)
            )
            
            # Load weights and training history
            model_instance.weights = model_data['weights']
            model_instance.objective_func_vals = model_data.get('objective_func_vals', [])
            
            # Set the weights in the regressor
            model_instance.regressor.weights = model_data['weights']
            
            print(f"✓ Model loaded from {file_path}")
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
            plt.plot(self.objective_func_vals, 'b-', linewidth=2, marker='o', 
                    markersize=4, label="Objective Function")
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Objective Function Value", fontsize=12)
            plt.title("Quantum Regressor Training Progress (EstimatorQNN)", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Training curve saved to {save_path}")
            
        except Exception as e:
            print(f"⚠ Failed to save training curve: {e}")
    
    def print_model(self, file_name: str = "quantum_regressor_circuit_estimator.png"):
        """
        Display and save the quantum circuit diagram with model information.
        
        Args:
            file_name (str, optional): Filename to save the circuit diagram.
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
            print(f"✓ Circuit image saved as {file_name}")
            
            # Print model information
            print(f"\n{'='*60}")
            print(f"Quantum Regressor Information (EstimatorQNN)")
            print(f"{'='*60}")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Circuit depth: {self.qc.depth()}")
            print(f"Number of parameters: {self.qc.num_parameters}")
            print(f"Maximum iterations: {self.maxiter}")
            if self.weights is not None:
                print(f"Model Weights Shape: {self.weights.shape}")
                print(f"Weights range: [{self.weights.min():.4f}, {self.weights.max():.4f}]")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"⚠ Error displaying quantum circuit: {e}")


class QuantumRegressor_VQR_CPU:
    """
    Enhanced variational quantum regressor using VQR with improved training.
    """
    
    def __init__(self, num_qubits: int, maxiter: int = 50):
        """
        Initialize the Variational Quantum Regressor.
        
        Args:
            num_qubits (int): Number of qubits to use in the quantum circuit
            maxiter (int, optional): Maximum optimization iterations. Defaults to 50.
        """
        try:
            from qiskit_machine_learning.optimizers import COBYLA
            from qiskit_machine_learning.utils import algorithm_globals
            from qiskit_machine_learning.algorithms.regressors import VQR
            from qiskit.primitives import StatevectorEstimator as Estimator
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
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
            self._iteration_count = 0
            
            # Initialize quantum components with better circuits
            self.estimator = Estimator()
            
            # Use ZZFeatureMap for better feature encoding
            self.feature_map = ZZFeatureMap(num_qubits, reps=2)
            
            # Use RealAmplitudes ansatz for better expressivity
            self.ansatz = RealAmplitudes(num_qubits, reps=3)
            
            # Use COBYLA optimizer (more reliable for VQR)
            # Use COBYLA optimizer (more reliable for VQR)
            self.optimizer = COBYLA(maxiter=maxiter)
            
            self.regressor = VQR(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer,
                callback=self._callback_graph,
                estimator=self.estimator,
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize variational quantum regressor: {e}")
    
    def _callback_graph(self, weights: np.ndarray, obj_func_eval: float):
        """
        Callback function to track objective function during training.
        
        Args:
            weights (np.ndarray): Current model weights during training
            obj_func_eval (float): Current objective function value
        """
        self._iteration_count += 1
        self.objective_func_vals.append(obj_func_eval)
        
        # Update progress bar if available
        if hasattr(self, '_progress_bar') and self._progress_bar is not None:
            self._progress_bar.set_postfix(
                obj_func=f"{obj_func_eval:.6f}",
                iteration=self._iteration_count
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
                print(f"Optimizer: COBYLA, Max iterations: {self.maxiter}")
                print(f"Note: Training may take a few minutes...\n")
                
                # Create progress bar for training iterations
                self._progress_bar = tqdm(total=self.maxiter, desc="Training", 
                                        unit="iter", leave=True,
                                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                self._progress_bar = None
            
            # Reset objective function values and iteration count
            self.objective_func_vals = []
            self._iteration_count = 0
            
            # Train the regressor
            self.regressor.fit(X, y)
            self.weights = self.regressor.weights
            
            if verbose:
                if self._progress_bar is not None:
                    # Update progress bar to show completion
                    remaining = self.maxiter - self._iteration_count
                    if remaining > 0:
                        self._progress_bar.update(remaining)
                    self._progress_bar.close()
                
                # Display results
                if self.objective_func_vals:
                    print(f"\n✓ Training completed!")
                    print(f"  Total iterations: {len(self.objective_func_vals)}")
                    print(f"  Final objective: {self.objective_func_vals[-1]:.6f}")
                    print(f"  Initial objective: {self.objective_func_vals[0]:.6f}")
                    print(f"  Improvement: {self.objective_func_vals[0] - self.objective_func_vals[-1]:.6f}")
                    self._plot_training_curve()
                else:
                    print(f"\n✓ Training completed!")
                    print(f"  (Callback data not available - this is normal for some optimizers)")
            
            return self.weights
            
        except Exception as e:
            if hasattr(self, '_progress_bar') and self._progress_bar is not None:
                self._progress_bar.close()
            raise RuntimeError(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained variational quantum regressor."""
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
        """Evaluate the model's performance using R² score."""
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
        """Save the trained model to disk."""
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
            
            print(f"✓ Model saved to {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, file_path: str = "variational_quantum_regressor.pkl"):
        """Load a trained model from disk."""
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
            
            print(f"✓ Model loaded from {file_path}")
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
            plt.plot(self.objective_func_vals, 'b-', linewidth=2, marker='o',
                    markersize=4, label="Objective Function")
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Objective Function Value", fontsize=12)
            plt.title("Variational Quantum Regressor Training Progress", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Training curve saved to {save_path}")
            
        except Exception as e:
            print(f"⚠ Failed to save training curve: {e}")

    def print_model(self, file_name: str = "variational_quantum_regressor_circuit.png"):
        """Display and save the quantum circuit used in the model."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            # Save circuit diagram (feature map + ansatz)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=300)
            
            # Draw feature map
            self.feature_map.decompose().draw(output='mpl', ax=ax1, style='iqp')
            ax1.set_title("Feature Map (ZZFeatureMap)", fontsize=14, fontweight='bold')
            
            # Draw ansatz
            self.ansatz.decompose().draw(output='mpl', ax=ax2, style='iqp')
            ax2.set_title("Ansatz (RealAmplitudes)", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Circuit diagram saved as {file_name}")
            
            # Print model information
            print(f"\n{'='*60}")
            print(f"Variational Quantum Regressor Information")
            print(f"{'='*60}")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Feature map depth: {self.feature_map.depth()}")
            print(f"Ansatz depth: {self.ansatz.depth()}")
            print(f"Total parameters: {self.ansatz.num_parameters}")
            print(f"Maximum iterations: {self.maxiter}")
            if self.weights is not None:
                print(f"Model Weights Shape: {self.weights.shape}")
                print(f"Weights range: [{self.weights.min():.4f}, {self.weights.max():.4f}]")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"⚠ Error displaying quantum circuit: {e}")