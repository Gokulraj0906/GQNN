"""
Quantum Support Vector Machine (QSVM) Models

This module provides implementations of QSVM for both classification and regression
tasks using Qiskit and Qiskit Machine Learning.

Classes:
    - QSVC_CPU: Quantum Support Vector Classifier
    - QSVR_CPU: Quantum Support Vector Regressor
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class QSVC_CPU:
    """
    Quantum Support Vector Classifier using quantum kernel methods.
    
    This classifier uses a quantum feature map to compute kernel matrices,
    which are then used in a classical SVM for classification tasks.
    
    Attributes:
        num_qubits (int): Number of qubits in the quantum circuit
        feature_map: Quantum circuit for encoding classical data
        quantum_kernel: Quantum kernel for computing similarities
        qsvc: Quantum Support Vector Classifier instance
        training_time (float): Time taken for training
    """
    
    def __init__(self, num_qubits: int, feature_map_reps: int = 2):
        """
        Initialize the Quantum Support Vector Classifier.
        
        Args:
            num_qubits (int): Number of qubits (must match number of features)
            feature_map_reps (int, optional): Repetitions in feature map. Defaults to 2.
            
        Raises:
            ImportError: If required Qiskit packages are not installed
            ValueError: If input parameters are invalid
        """
        try:
            from qiskit_machine_learning.algorithms import QSVC
            from qiskit_machine_learning.kernels import FidelityQuantumKernel
            from qiskit.circuit.library import ZZFeatureMap
            from qiskit_machine_learning.state_fidelities import ComputeUncompute
            from qiskit.primitives import StatevectorSampler  # FIXED: Use StatevectorSampler
            from qiskit_machine_learning.utils import algorithm_globals
        except ImportError as e:
            raise ImportError(
                f"Required package not found: {e}. "
                "Please install qiskit-machine-learning and qiskit-algorithms."
            )
        
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if feature_map_reps <= 0:
            raise ValueError("feature_map_reps must be positive")
        
        try:
            # Set random seed for reproducibility
            algorithm_globals.random_seed = 42
            
            self.num_qubits = num_qubits
            self.feature_map_reps = feature_map_reps
            self.training_time = 0
            
            # Create quantum feature map (ZZFeatureMap for better entanglement)
            self.feature_map = ZZFeatureMap(
                feature_dimension=num_qubits,
                reps=feature_map_reps,
                entanglement='linear'
            )
            
            # Create quantum kernel with proper fidelity
            sampler = StatevectorSampler()  # FIXED: Instantiate StatevectorSampler
            fidelity = ComputeUncompute(sampler=sampler)
            self.quantum_kernel = FidelityQuantumKernel(
                feature_map=self.feature_map,
                fidelity=fidelity
            )
            
            # Create QSVC instance
            self.qsvc = QSVC(quantum_kernel=self.quantum_kernel)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QSVC: {e}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Train the quantum support vector classifier.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target labels of shape (n_samples,)
            verbose (bool, optional): Whether to display progress. Defaults to True.
            
        Returns:
            self: Fitted classifier instance
        """
        try:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if X.shape[1] != self.num_qubits:
                raise ValueError(
                    f"Number of features ({X.shape[1]}) must match "
                    f"number of qubits ({self.num_qubits})"
                )
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            if verbose:
                print(f"Training QSVC with {len(X)} samples...")
                print(f"Features: {X.shape[1]}, Qubits: {self.num_qubits}")
                print(f"Classes: {np.unique(y)}")
                print("Note: Computing quantum kernel matrix...\n")
            
            import time
            start_time = time.time()
            
            # Fit the classifier
            if verbose:
                with tqdm(total=1, desc="Training", unit="step") as pbar:
                    self.qsvc.fit(X, y)
                    pbar.update(1)
            else:
                self.qsvc.fit(X, y)
            
            self.training_time = time.time() - start_time
            
            if verbose:
                print(f"\n✓ Training completed in {self.training_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels
        """
        try:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if X.shape[1] != self.num_qubits:
                raise ValueError(
                    f"Number of features ({X.shape[1]}) must match "
                    f"number of qubits ({self.num_qubits})"
                )
            
            predictions = self.qsvc.predict(X)
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X (np.ndarray): Test features of shape (n_samples, n_features)
            y (np.ndarray): True labels of shape (n_samples,)
            
        Returns:
            float: Mean accuracy score
        """
        try:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            accuracy = self.qsvc.score(X, y)
            return accuracy
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def save_model(self, file_path: str = "qsvc_model.pkl"):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str, optional): Path to save the model.
        """
        try:
            model_data = {
                'num_qubits': self.num_qubits,
                'feature_map_reps': self.feature_map_reps,
                'training_time': self.training_time,
                'qsvc': self.qsvc
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✓ Model saved to {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, file_path: str = "qsvc_model.pkl"):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str, optional): Path to the saved model.
            
        Returns:
            QSVC_CPU: Loaded model instance
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model_instance = cls(
                num_qubits=model_data['num_qubits'],
                feature_map_reps=model_data['feature_map_reps']
            )
            
            model_instance.qsvc = model_data['qsvc']
            model_instance.training_time = model_data.get('training_time', 0)
            
            print(f"✓ Model loaded from {file_path}")
            return model_instance
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def print_model(self, file_name: str = "qsvc_feature_map.png"):
        """
        Display and save the quantum feature map circuit.
        
        Args:
            file_name (str, optional): Filename to save the circuit diagram.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            
            fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
            self.feature_map.decompose().draw(output='mpl', ax=ax, style='iqp')
            plt.title("QSVC Feature Map (ZZFeatureMap)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Circuit diagram saved as {file_name}")
            
            print(f"\n{'='*60}")
            print(f"Quantum SVC Information")
            print(f"{'='*60}")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Feature map depth: {self.feature_map.depth()}")
            print(f"Feature map parameters: {self.feature_map.num_parameters}")
            print(f"Feature map reps: {self.feature_map_reps}")
            if self.training_time > 0:
                print(f"Training time: {self.training_time:.2f} seconds")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"⚠ Error displaying circuit: {e}")


class QSVR_CPU:
    """
    Quantum Support Vector Regressor using quantum kernel methods.
    
    This regressor uses a quantum feature map to compute kernel matrices,
    which are then used in a classical SVR for regression tasks.
    
    Attributes:
        num_qubits (int): Number of qubits in the quantum circuit
        feature_map: Quantum circuit for encoding classical data
        quantum_kernel: Quantum kernel for computing similarities
        qsvr: Quantum Support Vector Regressor instance
        training_time (float): Time taken for training
    """
    
    def __init__(self, num_qubits: int, feature_map_reps: int = 2, epsilon: float = 0.1):
        """
        Initialize the Quantum Support Vector Regressor.
        
        Args:
            num_qubits (int): Number of qubits (must match number of features)
            feature_map_reps (int, optional): Repetitions in feature map. Defaults to 2.
            epsilon (float, optional): Epsilon in epsilon-SVR model. Defaults to 0.1.
            
        Raises:
            ImportError: If required Qiskit packages are not installed
            ValueError: If input parameters are invalid
        """
        try:
            from qiskit_machine_learning.algorithms import QSVR
            from qiskit_machine_learning.kernels import FidelityQuantumKernel
            from qiskit.circuit.library import ZZFeatureMap
            from qiskit_machine_learning.state_fidelities import ComputeUncompute
            from qiskit.primitives import StatevectorSampler  # FIXED: Use StatevectorSampler
            from qiskit_machine_learning.utils import algorithm_globals
        except ImportError as e:
            raise ImportError(
                f"Required package not found: {e}. "
                "Please install qiskit-machine-learning and qiskit-algorithms."
            )
        
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if feature_map_reps <= 0:
            raise ValueError("feature_map_reps must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        
        try:
            # Set random seed for reproducibility
            algorithm_globals.random_seed = 42
            
            self.num_qubits = num_qubits
            self.feature_map_reps = feature_map_reps
            self.epsilon = epsilon
            self.training_time = 0
            
            # Create quantum feature map
            self.feature_map = ZZFeatureMap(
                feature_dimension=num_qubits,
                reps=feature_map_reps,
                entanglement='linear'
            )
            
            # Create quantum kernel with proper fidelity
            sampler = StatevectorSampler()  # FIXED: Instantiate StatevectorSampler
            fidelity = ComputeUncompute(sampler=sampler)  # FIXED: Create ComputeUncompute fidelity
            self.quantum_kernel = FidelityQuantumKernel(
                feature_map=self.feature_map,
                fidelity=fidelity  # FIXED: Pass fidelity, not sampler
            )
            
            # Create QSVR instance
            self.qsvr = QSVR(quantum_kernel=self.quantum_kernel, epsilon=epsilon)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QSVR: {e}")

    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Train the quantum support vector regressor.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
            verbose (bool, optional): Whether to display progress. Defaults to True.
            
        Returns:
            self: Fitted regressor instance
        """
        try:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if X.shape[1] != self.num_qubits:
                raise ValueError(
                    f"Number of features ({X.shape[1]}) must match "
                    f"number of qubits ({self.num_qubits})"
                )
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            # Flatten y if needed
            y = np.array(y).flatten()
            
            if verbose:
                print(f"Training QSVR with {len(X)} samples...")
                print(f"Features: {X.shape[1]}, Qubits: {self.num_qubits}")
                print(f"Epsilon: {self.epsilon}")
                print("Note: Computing quantum kernel matrix...\n")
            
            import time
            start_time = time.time()
            
            # Fit the regressor
            if verbose:
                with tqdm(total=1, desc="Training", unit="step") as pbar:
                    self.qsvr.fit(X, y)
                    pbar.update(1)
            else:
                self.qsvr.fit(X, y)
            
            self.training_time = time.time() - start_time
            
            if verbose:
                print(f"\n✓ Training completed in {self.training_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted target values
        """
        try:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if X.shape[1] != self.num_qubits:
                raise ValueError(
                    f"Number of features ({X.shape[1]}) must match "
                    f"number of qubits ({self.num_qubits})"
                )
            
            predictions = self.qsvr.predict(X)
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R² of the prediction.
        
        Args:
            X (np.ndarray): Test features of shape (n_samples, n_features)
            y (np.ndarray): True target values of shape (n_samples,)
            
        Returns:
            float: R² score
        """
        try:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if len(X) != len(y):
                raise ValueError("X and y must have same number of samples")
            
            y = np.array(y).flatten()
            r2_score = self.qsvr.score(X, y)
            return r2_score
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def save_model(self, file_path: str = "qsvr_model.pkl"):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str, optional): Path to save the model.
        """
        try:
            model_data = {
                'num_qubits': self.num_qubits,
                'feature_map_reps': self.feature_map_reps,
                'epsilon': self.epsilon,
                'training_time': self.training_time,
                'qsvr': self.qsvr
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✓ Model saved to {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, file_path: str = "qsvr_model.pkl"):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str, optional): Path to the saved model.
            
        Returns:
            QSVR_CPU: Loaded model instance
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model_instance = cls(
                num_qubits=model_data['num_qubits'],
                feature_map_reps=model_data['feature_map_reps'],
                epsilon=model_data['epsilon']
            )
            
            model_instance.qsvr = model_data['qsvr']
            model_instance.training_time = model_data.get('training_time', 0)
            
            print(f"✓ Model loaded from {file_path}")
            return model_instance
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def print_model(self, file_name: str = "qsvr_feature_map.png"):
        """
        Display and save the quantum feature map circuit.
        
        Args:
            file_name (str, optional): Filename to save the circuit diagram.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            
            fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
            self.feature_map.decompose().draw(output='mpl', ax=ax, style='iqp')
            plt.title("QSVR Feature Map (ZZFeatureMap)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Circuit diagram saved as {file_name}")
            
            print(f"\n{'='*60}")
            print(f"Quantum SVR Information")
            print(f"{'='*60}")
            print(f"Number of qubits: {self.num_qubits}")
            print(f"Feature map depth: {self.feature_map.depth()}")
            print(f"Feature map parameters: {self.feature_map.num_parameters}")
            print(f"Feature map reps: {self.feature_map_reps}")
            print(f"Epsilon: {self.epsilon}")
            if self.training_time > 0:
                print(f"Training time: {self.training_time:.2f} seconds")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"⚠ Error displaying circuit: {e}")