import matplotlib
matplotlib.use("Agg")
matplotlib.use("TkAgg")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from GQNN.models.classification_model import (
    QuantumClassifier_EstimatorQNN_CPU,
    QuantumClassifier_SamplerQNN_CPU,
    VariationalQuantumClassifier_CPU
)

# Data prep
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Helper to run, evaluate, save, visualize
def run_model(model, name):
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train, verbose=True)
    acc = model.score(X_test, y_test)
    print(f"{name} Accuracy: {acc:.4f}")
    model.save_model(f"{name.lower()}.pkl")
    model.print_model(f"{name.lower()}_circuit.png")

# Run different models
run_model(
    QuantumClassifier_EstimatorQNN_CPU(num_qubits=2, batch_size=32, lr=0.001),
    "EstimatorQNN"
)

run_model(
    QuantumClassifier_SamplerQNN_CPU(num_inputs=2, output_shape=2, ansatz_reps=1, maxiter=50),
    "SamplerQNN"
)

run_model(
    VariationalQuantumClassifier_CPU(num_inputs=2, maxiter=30),
    "VariationalQNN"
)
