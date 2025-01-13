
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
    
    
    def __init__(self, num_qubits: int, maxiter: int, random_seed: int):
        """
        Initializes the QuantumClassifier with the specified parameters.
        
        Args:
            num_qubits (int): The number of qubits in the quantum circuit.
            maxiter (int): The maximum number of iterations for the optimizer.
            random_seed (int): The random seed used for reproducibility.
        """
        from qiskit_machine_learning.optimizers import COBYLA
        from qiskit_machine_learning.utils import algorithm_globals
        from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.circuit.library import QNNCircuit
        from qiskit.primitives import StatevectorEstimator as Estimator

        algorithm_globals.random_seed = random_seed

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

    def print_model(self):
        """
        Returns the quantum circuit and the model's learned weights.
        
        This method draws the quantum circuit and returns it, along with the model's weights.
        """
        import matplotlib.pyplot as plt
        if hasattr(self, 'qc') and self.qc is not None:
            try:
                plt.figure(figsize=(10, 8))  # Set figure size for better readability
                plt.title("Quantum Circuit")
                self.qc.draw('mpl')  # Draw the quantum circuit using Matplotlib
                plt.show()  # Display the plot
                plt.savefig('Quantum_Circuit.png')  # Save the plot as a PNG file
            except Exception as e:
                print(f"Error displaying quantum circuit: {e}")
        else:
            print("Quantum circuit is not initialized.")

        print("Quantum Neural Network Model:")
        print(self.qc)
        print("Model Weights: ", self.weights)