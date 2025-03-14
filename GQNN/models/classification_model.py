import torch
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer

class QuantumClassifier_EstimatorQNN_CPU:
    def __init__(self, num_qubits: int, maxiter: int = 30):
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.circuit.library import QNNCircuit
        from qiskit.primitives import StatevectorEstimator as Estimator
        from qiskit_machine_learning.connectors import TorchConnector

        self.qc = QNNCircuit(num_qubits)
        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(circuit=self.qc, estimator=self.estimator)
        self.model = TorchConnector(self.estimator_qnn)

        # ✅ Use a faster optimizer (SGD)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.5, momentum=0.9)

        # ✅ Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y, epochs=15):  # ✅ Fewer epochs
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        loss_fn = torch.nn.MSELoss()
        loss_history = []

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = loss_fn(output, y_tensor)
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # ✅ Plot loss after training (not during)
        plt.plot(range(epochs), loss_history, 'b-o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.savefig("training_loss.png")
        plt.close()
        print("Training loss graph saved as 'training_loss.png'.")

    def print_quantum_circuit(self):
        print(self.qc)
        circuit_drawer(self.qc.decompose(), output='mpl', filename="quantum_circuit.png")
        print("Quantum circuit diagram saved as 'quantum_circuit.png'.")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions.round() == y).mean()
        return accuracy



""""This code will runs on Local computer """

class QuantumClassifier_SamplerQNN_CPU:
    def __init__(self, num_inputs:int, output_shape:None|int = 2, ansatz_reps:int|int = 1, maxiter:int|int=30):
        """
        Initialize the QuantumClassifier with customizable parameters.

        Args:
            num_inputs (int): Number of inputs for the feature map and ansatz.
            output_shape (int): Number of output classes for the QNN.
            ansatz_reps (int): Number of repetitions for the ansatz circuit.
            random_seed (int, optional): Seed for random number generation.
        """
        from qiskit.circuit.library import RealAmplitudes
        from qiskit_machine_learning.optimizers import COBYLA
        from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit_machine_learning.circuit.library import QNNCircuit
        from qiskit.primitives import StatevectorSampler
        self.num_inputs = num_inputs
        self.output_shape = output_shape
        self.ansatz_reps = ansatz_reps
        self.sampler = StatevectorSampler()
        self.objective_func_vals = []
        self.qnn_circuit = QNNCircuit(ansatz=RealAmplitudes(self.num_inputs, reps=self.ansatz_reps))
        self.qnn = SamplerQNN(
            circuit=self.qnn_circuit,
            interpret=self.parity,
            output_shape=self.output_shape,
            sampler=self.sampler,
        )
        self.classifier = NeuralNetworkClassifier(
            neural_network=self.qnn,
            optimizer=COBYLA(maxiter=maxiter),
            callback=self._callback_graph
        )

    @staticmethod
    def parity(x):
        """
        Interpret the binary parity of the input.

        Args:
            x (int): Input integer.

        Returns:
            int: Parity of the input.
        """
        return "{:b}".format(x).count("1") % 2

    def _callback_graph(self, weights, obj_func_eval):
        """
        Callback to update the objective function graph during training.

        This method is called during training to update the objective function plot and save it as an image.
        
        Args:
            weights (numpy.ndarray): The weights of the model during training.
            obj_func_eval (float): The value of the objective function at the current iteration.
        """
        from IPython.display import clear_output
        import matplotlib.pyplot as plt
        import warnings
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
        Fit the classifier to the provided data.

        Args:
            X (ndarray): Training features.
            y (ndarray): Training labels.
        """
        import matplotlib.pyplot as plt
        plt.ion()
        self.classifier.fit(X, y)
        self.weights = self.classifier.weights
        plt.ioff()
        plt.show()

    def score(self, X, y):
        """
        Evaluate the classifier on the provided data.

        Args:
            X (ndarray): Features for evaluation.
            y (ndarray): Labels for evaluation.

        Returns:
            float: Accuracy score.
        """
        return self.classifier.score(X, y)

    def print_model(self,file_name="quantum_circuit.png"):
        """
        Display the quantum circuit and save it as an image.

        This method uses Matplotlib to render the quantum circuit and saves the plot.
        """
        try:
            circuit = self.qnn_circuit.decompose().draw(output='mpl')
            circuit.savefig(file_name)
            print(f"Circuit image saved as {file_name}")
        except Exception as e:
            print(f"Error displaying or saving the quantum circuit: {e}")

        print("Quantum Circuit:")
        print(self.qnn_circuit)
        print("Model Weights:", self.classifier.weights)

""""This code will runs on Local computer """

class VariationalQuantumClassifier_CPU:
    """
    A class for building, training, and evaluating a Variational Quantum Classifier (VQC).

    Attributes:
        num_inputs (int): Number of qubits/features in the quantum circuit.
        max_iter (int): Maximum iterations for the optimizer.
        feature_map (QuantumCircuit): Feature map used for embedding classical data into a quantum state.
        ansatz (QuantumCircuit): Ansatz used as the variational component of the quantum circuit.
        sampler (Sampler): Backend for quantum computations.
        vqc (VQC): The Variational Quantum Classifier model.
        objective_func_vals (list): List to store objective function values during training.
    """

    def __init__(self, num_inputs: int = 2, max_iter: int = 30):
        """
        Initialize the VQC with a feature map, ansatz, and optimizer.
        
        Args:
            num_inputs (int): Number of qubits/features.
            max_iter (int): Maximum iterations for the optimizer.
        """
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from qiskit_machine_learning.algorithms.classifiers import VQC
        from qiskit_machine_learning.optimizers import COBYLA
        from qiskit.primitives import StatevectorSampler

        self.num_inputs = num_inputs
        self.max_iter = max_iter
        self.objective_func_vals = []
        
        # Initialize feature map, ansatz, and sampler
        self.feature_map = ZZFeatureMap(num_inputs)
        self.ansatz = RealAmplitudes(num_inputs, reps=1)
        self.sampler = StatevectorSampler()
        
        # Initialize VQC model
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            loss="cross_entropy",
            optimizer=COBYLA(maxiter=self.max_iter),
            callback=self._callback_graph,
            sampler=self.sampler,
        )

    def _callback_graph(self, weights, obj_func_eval):
        """
        Callback function to visualize the objective function value during training.
        
        Args:
            weights (np.ndarray): Model weights during training.
            obj_func_eval (float): Current objective function value.
        """
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="FigureCanvasAgg is non-interactive")
        clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        plt.title("Objective Function Value During Training")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals, color='b')
        plt.show()
        plt.savefig("Training Graph.png")

    import numpy as np
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the VQC on the provided dataset.
        
        Args:
            X (np.ndarray): Training data (features).
            y (np.ndarray): Training data (labels).
        """
        import numpy as np
        y = np.array(y)
        import matplotlib.pyplot as plt
        plt.ion()  # Enable interactive mode for live plotting
        self.vqc.fit(X, y)
        self.weights = self.vqc.weights
        plt.ioff()  # Disable interactive mode
        plt.show()

    import numpy as np
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the input data.
        
        Args:
            X (np.ndarray): Input data for prediction.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        return self.vqc.predict(X)
    import numpy as np
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the accuracy of the VQC on the provided dataset.
        
        Args:
            X (np.ndarray): Test data (features).
            y (np.ndarray): True labels.
        
        Returns:
            float: Accuracy score.
        """
        return self.vqc.score(X, y)

    def print_model(self, file_name: str = "quantum_circuit.png"):
        """
        Visualize and save the quantum circuit diagram.
        
        Args:
            file_name (str): File name to save the circuit diagram.
        """
        try:
            circuit = self.feature_map.compose(self.ansatz).decompose()
            circuit.draw(output="mpl").savefig(file_name)
            print(f"Circuit diagram saved as {file_name}")
        except Exception as e:
            print(f"Error visualizing the circuit: {e}")
        
        print("Quantum Circuit:")
        print(self.feature_map)
        print(self.ansatz)
        print("Model Weights:")
        print(self.vqc.weights)