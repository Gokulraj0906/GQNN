import matplotlib
matplotlib.use("Agg")
matplotlib.use("TkAgg")

from GQNN.models.regression_model import QuantumRegressor_EstimatorQNN_CPU
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate regression data
X, y = make_regression(n_samples=70, n_features=3, noise=0.1, random_state=42)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Initialize and train model
model = QuantumRegressor_EstimatorQNN_CPU(
    num_qubits=3,           # Number of qubits
    maxiter=5             # Maximum iterations
)

# Train with progress tracking
weights = model.fit(X_train_scaled, y_train_scaled, verbose=True)

# Evaluate
r2_score = model.score(X_test_scaled, y_test_scaled)
print(f"R² Score: {r2_score:.4f}")

# Make predictions
predictions = model.predict(X_test_scaled)

# Save and visualize
model.save_model("quantum_regressor.pkl")
model.print_model("regressor_circuit.png")