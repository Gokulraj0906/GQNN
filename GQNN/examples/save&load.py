from GQNN.models.regression_model import QuantumRegressor_EstimatorQNN_CPU as QER
# import pickle
# model = QER(num_qubits=4)
# model.load_model('D:\Projects\GQNN\EstimatorQNN_model.model')
# model.print_model()
# print("Loaded model weights:", model.weights)

# if model.weights is None:
#     print("Restoring weights manually...")
#     with open("EstimatorQNN_model_weights.pkl", "rb") as f:
#         model.weights = pickle.load(f)

# print("Loaded Model Weights:", model.weights)


# from joblib import load

# model = load('D:\Projects\GQNN\EstimatorQNN_model.model')
# print("Loaded model weights:", model.weights)
# print("Model Type:", type(model))
# print("Available Methods:", dir(model))  # Lists all available methods
# print(model.predict([[0.11, 1, 3, 4]]))  # Predict using the loaded

# if hasattr(model.neural_network, "summary"):
#     model.neural_network.summary()
# else:
#     print("No summary method available.")


# model = QER.load_model('D:\Projects\GQNN\EstimatorQNN_model.model')
# print("Type of loaded model:", type(model))  # Should be <class 'GQNN.models.RegressionModel'>

# if isinstance(model, str):
#     print("❌ ERROR: Model is a string instead of an object.")
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

feature_map = ZZFeatureMap(2)  # Reduce complexity
ansatz = RealAmplitudes(2, reps=1)  # Fewer reps reduce overfitting