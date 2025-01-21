class QuantumClassifier_SamplerQNN_CPU:
    def __init__(self, num_inputs=2, output_shape=2, ansatz_reps=1):
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
        from qiskit_machine_learning.utils import algorithm_globals
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
            optimizer=COBYLA(maxiter=30),
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

if __name__ == "__main__":
    import numpy as np
    classifier = QuantumClassifier_SamplerQNN_CPU(num_inputs=2, output_shape=2, ansatz_reps=10)
    from qiskit_machine_learning.utils import algorithm_globals
    # Generate example data
    X = 2 * algorithm_globals.random.random([10, classifier.num_inputs])
    y = 1 * (np.sum(X, axis=1) >= 0)

    print("Training Data:")
    print(X)
    # Fit the classifier
    classifier.fit(X, y)

    # Evaluate the classifier
    accuracy = classifier.score(X, y)
    print(f"Accuracy: {accuracy}")

    # Display the model and save the circuit as an image
    classifier.print_model()