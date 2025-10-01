import matplotlib
matplotlib.use("Agg")

from GQNN.models.regression_model import QuantumRegressor_VQR_CPU
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simpler problem for better learning
X, y = make_regression(
    n_samples=150,
    n_features=3,         # 3 features
    n_informative=3,
    noise=3.0,
    random_state=42,
    bias=0.0
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# VQR with better circuits
model = QuantumRegressor_VQR_CPU(
    num_qubits=3,
    maxiter=100  # VQR needs more iterations
)

print("Starting training...")
weights = model.fit(X_train_scaled, y_train_scaled, verbose=True)

# Evaluate
r2_train = model.score(X_train_scaled, y_train_scaled)
r2_test = model.score(X_test_scaled, y_test_scaled)

print(f"\n{'='*60}")
print(f"Training R² Score: {r2_train:.4f}")
print(f"Test R² Score: {r2_test:.4f}")
print(f"Overfitting gap: {r2_train - r2_test:.4f}")
print(f"{'='*60}")

# Fixed prediction display
predictions = model.predict(X_test_scaled[:5])
print("\nSample Predictions vs Actual:")
for i in range(5):
    pred = predictions[i] if predictions.ndim == 1 else predictions[i][0]
    actual = y_test_scaled[i]
    error = abs(pred - actual)
    print(f"  Sample {i+1}: Pred={pred:.4f}, Actual={actual:.4f}, Error={error:.4f}")

model.save_model("quantum_regressor_vqr.pkl")
model.print_model("regressor_circuit_vqr.png")