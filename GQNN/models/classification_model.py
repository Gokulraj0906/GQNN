
""""This code will runs on Local computer """

class QuantumClassifier_EstimatorQNN_CPU:
    """
    A quantum machine learning classifier that utilizes a quantum neural network (QNN) for classification tasks.
    
    This Model Will Runs on the Local Computer.

    This classifier uses a quantum circuit (QNNCircuit) as the model, and employs the COBYLA optimizer 
    to train the quantum model. The training process updates the objective function, which is visualized during 
    training via a callback method. The class provides methods for training, predicting, evaluating accuracy, 
    saving, and loading the model.

    Attributes:
        qc (QNNCircuit): Quantum circuit representing the quantum neural network.
        estimator (Estimator): Estimator for measuring the quantum states.
        estimator_qnn (EstimatorQNN): The quantum neural network that integrates the quantum circuit and estimator.
        optimizer (COBYLA): Optimizer used to train the quantum neural network.
        classifier (NeuralNetworkClassifier): The neural network classifier that performs the training and prediction.
        weights (numpy.ndarray): The weights of the trained model.
        objective_func_vals (list): List to store the objective function values during training.
    
    Methods:
        _callback_graph(weights, obj_func_eval):
            Callback method to visualize and update the objective function during training.
        
        fit(X, y):
            Trains the quantum classifier using the provided data (X, y).
        
        score(X, y):
            Evaluates the accuracy of the trained model on the provided data (X, y).
        
        predict(X):
            Predicts the labels for the input data (X).
        
        print_model():
            Prints the quantum circuit and the model weights.
        
        save_model(file_path='quantum_model_weights.npy'):
            Saves the model weights to a specified file.
        
        load_model(file_path='quantum_model_weights.npy'):
            Loads the model weights from a specified file.
    """
    
    
    def __init__(self, num_qubits: int, maxiter: int):
        """
        Initializes the QuantumClassifier with the specified parameters.
        
        Args:
            num_qubits (int): The number of qubits in the quantum circuit.
            maxiter (int): The maximum number of iterations for the optimizer.
        """
        from qiskit_machine_learning.optimizers import COBYLA
        from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.circuit.library import QNNCircuit
        from qiskit.primitives import StatevectorEstimator as Estimator


        # Initialize quantum circuit, estimator, and neural network
        self.qc = QNNCircuit(num_qubits)
        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(circuit=self.qc, estimator=self.estimator)

        # Initialize optimizer and classifier
        self.optimizer = COBYLA(maxiter=maxiter)
        self.classifier = NeuralNetworkClassifier(self.estimator_qnn, optimizer=self.optimizer, callback=self._callback_graph)
        self.weights = None

        # Store objective function values for visualization during training
        self.objective_func_vals = []
    
 
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
        Trains the quantum classifier on the provided data.
        
        This method trains the model by fitting it to the input features (X) and labels (y).
        
        Args:
            X (numpy.ndarray): The input feature data for training.
            y (numpy.ndarray): The labels corresponding to the input features.
        """
        import matplotlib.pyplot as plt
        plt.ion()  # Enable interactive mode for live plotting
        self.classifier.fit(X, y)
        self.weights = self.classifier.weights
        plt.ioff()  # Disable interactive mode after training
        plt.show()

    def score(self, X, y):
        """
        Evaluates the accuracy of the trained classifier.
        
        Args:
            X (numpy.ndarray): The input feature data for evaluation.
            y (numpy.ndarray): The true labels corresponding to the input features.
        
        Returns:
            float: The accuracy score of the model on the provided data.
        """
        return self.classifier.score(X, y)
    
    def predict(self, X):
        """
        Predicts the labels for the input data.
        
        Args:
            X (numpy.ndarray): The input feature data to predict labels for.
        
        Returns:
            numpy.ndarray: The predicted labels for the input data.
        """
        if self.weights is None:
            raise ValueError("Model weights are not loaded or trained.")
        return self.classifier.predict(X)

    def print_model(self,file_name="quantum_circuit.png"):
        """
        Returns the quantum circuit and the model's learned weights.
        
        This method draws the quantum circuit and returns it, along with the model's weights.
        """
        import matplotlib.pyplot as plt
        if hasattr(self, 'qc') and self.qc is not None:
            try:
        
                circuit = self.qc.decompose().draw(output='mpl')
                circuit.savefig(file_name)
                print(f"Circuit image saved as {file_name}") 
            except Exception as e:
                print(f"Error displaying quantum circuit: {e}")
        else:
            print("Quantum circuit is not initialized.")

        print("Quantum Neural Network Model:")
        print(self.qc)
        print("Model Weights: ", self.weights)


""""This code will runs on Local computer """

class QuantumClassifier_SamplerQNN_CPU:
    def __init__(self, num_inputs:int, output_shape:int, ansatz_reps:int, maxiter:int):
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