""""This code will runs on Local computer """

class QuantumRegressor_EstimatorQNN_CPU:
    """
    A quantum machine learning regressor that utilizes a quantum neural network (QNN) for regression tasks.
    
    This Model Will Run on the Local Computer.

    This regressor uses a quantum circuit (QNNCircuit) as the model and employs the COBYLA optimizer
    to train the quantum model. The training process updates the objective function, which is visualized during
    training via a callback method. The class provides methods for training, predicting, evaluating performance,
    saving, and loading the model.
    
    Attributes:
        qc (QNNCircuit): Quantum circuit representing the quantum neural network.
        estimator (Estimator): Estimator for measuring the quantum states.
        estimator_qnn (EstimatorQNN): The quantum neural network that integrates the quantum circuit and estimator.
        optimizer (COBYLA): Optimizer used to train the quantum neural network.
        regressor (NeuralNetworkRegressor): The neural network regressor that performs training and prediction.
        weights (numpy.ndarray): The weights of the trained model.
        objective_func_vals (list): List to store the objective function values during training.
    
    Methods:
        _callback_graph(weights, obj_func_eval):
            Callback method to visualize and update the objective function during training.
        
        fit(X, y):
            Trains the quantum regressor using the provided data (X, y).
        
        score(X, y):
            Evaluates the performance of the trained model on the provided data (X, y).
        
        predict(X):
            Predicts the labels for the input data (X).
        
        print_model(file_name="quantum_circuit.png"):
            Saves the quantum circuit as an image and prints the model weights.
    """
    
    def __init__(self, num_qubits: int, maxiter: int = 30):
        """
        Initializes the QuantumRegressor with the specified parameters.
        
        Args:
            num_qubits (int): The number of qubits in the quantum circuit.
            maxiter (int): The maximum number of iterations for the optimizer.
        """
        from qiskit_machine_learning.optimizers import COBYLA
        from qiskit_machine_learning.utils import algorithm_globals
        from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.circuit.library import QNNCircuit
        from qiskit.primitives import StatevectorEstimator as Estimator
        
        algorithm_globals.random_seed = 42
        self.qc = QNNCircuit(num_qubits)
        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(circuit=self.qc, estimator=self.estimator)
        self.optimizer = COBYLA(maxiter=maxiter)
        self.regressor = NeuralNetworkRegressor(
            neural_network=self.estimator_qnn,
            loss="squared_error",
            optimizer=self.optimizer,
            callback=self._callback_graph
        )
        self.weights = None
        self.objective_func_vals = []
    
    def _callback_graph(self, weights, obj_func_eval):
        """
        Callback to update the objective function graph during training.

        Args:
            weights (numpy.ndarray): The weights of the model during training.
            obj_func_eval (float): The value of the objective function at the current iteration.
        """
        from IPython.display import clear_output
        import warnings
        import matplotlib.pyplot as plt 
        warnings.filterwarnings("ignore", category=UserWarning, message="FigureCanvasAgg is non-interactive")
        clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        plt.title("Objective Function Value During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals, color='b')
        plt.show()
        plt.savefig('Training Graph.png')
    
    def fit(self, X, y):
        """
        Trains the quantum regressor on the provided data.
        
        Args:
            X (numpy.ndarray): The input feature data for training.
            y (numpy.ndarray): The target values corresponding to the input features.
        """
        import matplotlib.pyplot as plt
        plt.ion()
        self.regressor.fit(X, y)
        self.weights = self.regressor.weights
        plt.ioff()
        plt.show()
    
    def score(self, X, y):
        """
        Evaluates the performance of the trained regressor.
        
        Args:
            X (numpy.ndarray): The input feature data for evaluation.
            y (numpy.ndarray): The true target values corresponding to the input features.
        
        Returns:
            float: The R-squared score of the model on the provided data.
        """
        return self.regressor.score(X, y)
    
    def predict(self, X):
        """
        Predicts the output values for the input data.
        
        Args:
            X (numpy.ndarray): The input feature data to predict values for.
        
        Returns:
            numpy.ndarray: The predicted output values for the input data.
        """
        if self.weights is None:
            raise ValueError("Model weights are not loaded or trained.")
        return self.regressor.predict(X)
    
    def print_model(self, file_name="quantum_circuit.png"):
        """
        Saves the quantum circuit as an image and prints model details.
        
        Args:
            file_name (str): The filename to save the quantum circuit diagram.
        """
        import matplotlib.pyplot as plt
        if hasattr(self, 'qc') and self.qc is not None:
            try:
                fig, ax = plt.subplots()
                self.qc.decompose().draw(output='mpl', ax=ax)
                fig.savefig(file_name)
                print(f"Circuit image saved as {file_name}")
            except Exception as e:
                print(f"Error displaying quantum circuit: {e}")
        else:
            print("Quantum circuit is not initialized.")

        print("Quantum Neural Network Model:")
        print(self.qc)
        print("Model Weights: ", self.weights)

""""This code will runs on Local computer """

class QuantumRegressor_VQR_CPU:
    """
    A quantum variational regressor using Qiskit's Variational Quantum Regressor (VQR).
    This class implements a quantum circuit-based regression model that utilizes
    a feature map and ansatz to approximate a given function.

    Attributes:
        num_qubits (int): Number of qubits in the quantum circuit.
        maxiter (int): Maximum number of iterations for the optimizer.
        objective_func_vals (list): Stores the values of the objective function during training.
        estimator (Estimator): The quantum estimator used for circuit evaluation.
        feature_map (QuantumCircuit): The feature map encoding classical data into the quantum state.
        ansatz (QuantumCircuit): The ansatz circuit used to optimize parameters.
        optimizer (L_BFGS_B): Classical optimizer for training the quantum model.
        regressor (VQR): Variational Quantum Regressor instance.
        weights (np.ndarray or None): Learned weights of the trained quantum model.
    """
    
    def __init__(self, num_qubits: int = 1, maxiter: int = 5):
        """
        Initializes the Quantum Regressor with the given number of qubits and optimization iterations.

        Args:
            num_qubits (int): The number of qubits to use in the quantum circuit (default: 1).
            maxiter (int): The maximum number of optimization iterations (default: 5).
        """
        from qiskit_machine_learning.optimizers import L_BFGS_B
        from qiskit_machine_learning.utils import algorithm_globals
        from qiskit_machine_learning.algorithms.regressors import VQR
        from qiskit.primitives import StatevectorEstimator as Estimator
        algorithm_globals.random_seed = 42
        self.objective_func_vals = []
        self.estimator = Estimator()
        self.num_qubits = num_qubits
        self._initialize_circuit()
        self.optimizer = L_BFGS_B(maxiter=maxiter)
        self.regressor = VQR(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            callback=self._callback_graph,
            estimator=self.estimator,
        )
        self.weights = None
    
    def _initialize_circuit(self):
        """
        Initializes the quantum circuit with a feature map and an ansatz.
        The feature map encodes classical data into quantum states, and the ansatz
        is used for variational training.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        param_x = Parameter("x")
        self.feature_map = QuantumCircuit(self.num_qubits, name="FeatureMap")
        self.feature_map.ry(param_x, range(self.num_qubits))

        param_y = Parameter("y")
        self.ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")
        self.ansatz.ry(param_y, range(self.num_qubits))
    
    def _callback_graph(self, weights, obj_func_eval):
        """
        Callback function to visualize the objective function value over training iterations.
        
        Args:
            weights (np.ndarray): Current weights of the quantum model.
            obj_func_eval (float): Current objective function value.
        """       
        import matplotlib.pyplot as plt
        import numpy as np
        from IPython.display import clear_output
        clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        plt.title("Objective Function Value During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals, color='b')
        plt.show()
        plt.savefig('Training_Graph.png')
    
    def fit(self, X, y):
        """
        Trains the quantum regressor on the provided dataset.
        
        Args:
            X (np.ndarray): Training input data.
            y (np.ndarray): Target output values.
        """
        import matplotlib.pyplot as plt
        plt.ion()
        self.regressor.fit(X, y)
        self.weights = self.regressor.weights
        plt.ioff()
        plt.show()
    
    def score(self, X, y):
        """
        Evaluates the model's performance on the given dataset.
        
        Args:
            X (np.ndarray): Test input data.
            y (np.ndarray): True output values.
        
        Returns:
            float: The score of the model.
        """
        return self.regressor.score(X, y)
    
    def predict(self, X):
        """
        Predicts outputs using the trained quantum regressor.
        
        Args:
            X (np.ndarray): Input data for prediction.
        
        Returns:
            np.ndarray: Predicted output values.
        
        Raises:
            ValueError: If the model weights are not trained.
        """
        if self.weights is None:
            raise ValueError("Model weights are not loaded or trained.")
        return self.regressor.predict(X)

    def print_model(self, file_name="quantum_circuit.png"):
        """
        Prints and saves the quantum circuit used in the model.
        
        Args:
            file_name (str): Name of the file to save the circuit diagram (default: "quantum_circuit.png").
        """
        import matplotlib.pyplot as plt
        try:
            fig, ax = plt.subplots()
            self.feature_map.decompose().draw(output='mpl', ax=ax)
            fig.savefig(file_name)
            print(f"Circuit image saved as {file_name}")
        except Exception as e:
            print(f"Error displaying quantum circuit: {e}")
        print("Quantum Neural Network Model:")
        print(self.feature_map)
        print("Model Weights: ", self.weights)