from GQNN.models.classification_model import QuantumClassifier_SamplerQNN_CPU as model

model = model.load_model(file_path="quantum_model.pth")

model.print_quantum_circuit()