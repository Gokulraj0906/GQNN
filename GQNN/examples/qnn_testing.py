"""
Complete Example: Advanced Quantum Neural Network Framework
Binary Classification on Iris Dataset (Setosa vs. Versicolor)

This demonstrates all features of the QNN framework including:
- Multiple encoding strategies
- Different entanglement patterns
- Loss functions and regularization
- Callbacks (EarlyStopping, ModelCheckpoint)
- Cross-validation
- Batch processing
- Model saving/loading
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import your QNN framework classes here
# from quantum_nn_framework import *
from GQNN.models.qnn import EarlyStopping, ModelCheckpoint, QuantumNeuralNetwork_Basic_CPU
from GQNN.models.qnn import EncodingLayer, VariationalLayer, MeasurementLayer
# ==================== DATA PREPARATION ====================

def prepare_iris_binary_dataset():
    """Prepare binary classification dataset from Iris."""
    iris = load_iris()
    X = iris.data[:100, :2]  # Use only first two features
    y = iris.target[:100]     # Only first two classes

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale to [0, pi] for quantum encoding
    X_min, X_max = X_scaled.min(axis=0), X_scaled.max(axis=0)
    X_quantum = (X_scaled - X_min) / (X_max - X_min) * np.pi

    return train_test_split(X_quantum, y, test_size=0.2, random_state=42, stratify=y)


# ==================== EXAMPLE 1: BASIC USAGE ====================

def example_1_basic_training():
    """Example 1: Basic QNN training with default settings"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic QNN Training")
    print("="*80)

    X_train, X_test, y_train, y_test = prepare_iris_binary_dataset()

    # Create QNN with 2 qubits
    qnn = QuantumNeuralNetwork_Basic_CPU(n_qubits=2, name="BasicQNN", shots=1024)

    # Add layers
    qnn.add(EncodingLayer(n_qubits=2, encoding_type='angle', name='encoder'))
    qnn.add(VariationalLayer(n_qubits=2, n_layers=2, entanglement='linear', name='var_layer'))
    qnn.add(MeasurementLayer(n_qubits=2, observable='Z', name='measurement'))

    # Build and compile
    qnn.build()
    qnn.compile(loss='mse', regularization={'l1': 0.01, 'l2': 0.01})

    # Display model summary
    qnn.summary()

    # Train
    history = qnn.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        method='COBYLA',
        verbose=True
    )

    # Evaluate
    accuracy = qnn.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Save model and circuit
    qnn.save_model("basic_qnn_model.pkl")
    qnn.print_model("basic_qnn_circuit.png")

    return qnn, accuracy


# ==================== EXAMPLE 2: ADVANCED FEATURES ====================

def example_2_advanced_features():
    """Example 2: Using callbacks, cross-entropy loss, and complex architecture"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Advanced Features")
    print("="*80)

    X_train, X_test, y_train, y_test = prepare_iris_binary_dataset()

    # Create QNN with more layers
    qnn = QuantumNeuralNetwork_Basic_CPU(n_qubits=2, name="AdvancedQNN", shots=2048)

    # Multiple variational layers
    qnn.add(EncodingLayer(n_qubits=2, encoding_type='amplitude', name='amp_encoder'))
    qnn.add(VariationalLayer(n_qubits=2, n_layers=3, entanglement='circular', name='var1'))
    qnn.add(VariationalLayer(n_qubits=2, n_layers=2, entanglement='full', name='var2'))
    qnn.add(MeasurementLayer(n_qubits=2, observable='X', name='x_measurement'))

    qnn.build()
    qnn.compile(loss='cross_entropy', regularization={'l1': 0.005, 'l2': 0.01})
    qnn.summary()

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        min_delta=0.001,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath='best_qnn.pkl',
        monitor='val_loss',
        save_best_only=True
    )

    # Train with callbacks
    history = qnn.fit(
        X_train, y_train,
        validation_split=0.25,
        epochs=50,
        method='COBYLA',
        callbacks=[early_stopping, checkpoint],
        verbose=True
    )

    # Detailed evaluation
    accuracy, predictions = qnn.evaluate(X_test, y_test, return_predictions=True)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Predictions: {predictions}")
    print(f"True Labels: {y_test}")

    return qnn, accuracy


# ==================== EXAMPLE 3: CROSS-VALIDATION ====================

def example_3_cross_validation():
    """Example 3: K-fold cross-validation"""
    print("\n" + "="*80)
    print("EXAMPLE 3: K-Fold Cross-Validation")
    print("="*80)

    iris = load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_quantum = (X_scaled - X_scaled.min(axis=0)) / (X_scaled.max(axis=0) - X_scaled.min(axis=0)) * np.pi

    qnn = QuantumNeuralNetwork_Basic_CPU(n_qubits=2, name="CV_QNN", shots=1024)
    qnn.add(EncodingLayer(n_qubits=2, encoding_type='angle'))
    qnn.add(VariationalLayer(n_qubits=2, n_layers=2, entanglement='linear'))
    qnn.add(MeasurementLayer(n_qubits=2, observable='Z'))

    qnn.build()
    qnn.compile(loss='mse', regularization={'l2': 0.01})

    # 5-fold cross-validation
    cv_results = qnn.cross_validate(X_quantum, y, cv=5, verbose=True)

    print(f"\n{'='*80}")
    print("Cross-Validation Results:")
    print(f"  Fold scores: {cv_results['scores']}")
    print(f"  Mean ± Std: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    print(f"{'='*80}")

    return cv_results


# ==================== EXAMPLE 4: ENCODING COMPARISON ====================

def example_4_encoding_comparison():
    """Example 4: Compare different encoding strategies"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Encoding Strategy Comparison")
    print("="*80)

    X_train, X_test, y_train, y_test = prepare_iris_binary_dataset()
    encoding_types = ['angle', 'amplitude', 'iqp']
    results = {}

    for enc_type in encoding_types:
        print(f"\nTesting {enc_type.upper()} encoding...")

        qnn = QuantumNeuralNetwork_Basic_CPU(n_qubits=2, name=f"QNN_{enc_type}", shots=1024)
        qnn.add(EncodingLayer(n_qubits=2, encoding_type=enc_type))
        qnn.add(VariationalLayer(n_qubits=2, n_layers=2, entanglement='linear'))
        qnn.add(MeasurementLayer(n_qubits=2, observable='Z'))

        qnn.build()
        qnn.compile(loss='mse')
        qnn.fit(X_train, y_train, validation_split=0.2, epochs=20, verbose=False)

        accuracy = qnn.evaluate(X_test, y_test)
        results[enc_type] = accuracy
        print(f"  Accuracy: {accuracy:.4f}")

    print(f"\n{'='*80}")
    print("Summary:")
    for enc, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {enc.upper():<15}: {acc:.4f}")
    print(f"{'='*80}")

    return results


# ==================== EXAMPLE 5: ENTANGLEMENT COMPARISON ====================

def example_5_entanglement_comparison():
    """Example 5: Compare entanglement patterns"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Entanglement Pattern Comparison")
    print("="*80)

    X_train, X_test, y_train, y_test = prepare_iris_binary_dataset()
    entanglement_types = ['linear', 'circular', 'full', 'sca']
    results = {}

    for ent_type in entanglement_types:
        print(f"\nTesting {ent_type.upper()} entanglement...")

        qnn = QuantumNeuralNetwork_Basic_CPU(n_qubits=2, name=f"QNN_{ent_type}", shots=1024)
        qnn.add(EncodingLayer(n_qubits=2, encoding_type='angle'))
        qnn.add(VariationalLayer(n_qubits=2, n_layers=2, entanglement=ent_type))
        qnn.add(MeasurementLayer(n_qubits=2, observable='Z'))

        qnn.build()
        qnn.compile(loss='mse')

        depth = qnn.circuit.depth()
        gates = qnn.circuit.size()
        print(f"  Circuit - Depth: {depth}, Gates: {gates}")

        qnn.fit(X_train, y_train, validation_split=0.2, epochs=20, verbose=False)
        accuracy = qnn.evaluate(X_test, y_test)

        results[ent_type] = {'accuracy': accuracy, 'depth': depth, 'gates': gates}
        print(f"  Accuracy: {accuracy:.4f}")

    print(f"\n{'='*80}")
    print(f"{'Type':<15} {'Accuracy':<12} {'Depth':<10} {'Gates':<10}")
    print("-" * 50)
    for ent, metrics in results.items():
        print(f"{ent.upper():<15} {metrics['accuracy']:<12.4f} {metrics['depth']:<10} {metrics['gates']:<10}")
    print(f"{'='*80}")

    return results


# ==================== EXAMPLE 6: BATCH PREDICTION ====================

def example_6_batch_prediction():
    """Example 6: Batch vs mini-batch prediction"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Batch Prediction")
    print("="*80)

    X_train, X_test, y_train, y_test = prepare_iris_binary_dataset()

    qnn = QuantumNeuralNetwork_Basic_CPU(n_qubits=2, name="BatchQNN", shots=1024)
    qnn.add(EncodingLayer(n_qubits=2, encoding_type='angle'))
    qnn.add(VariationalLayer(n_qubits=2, n_layers=2, entanglement='linear'))
    qnn.add(MeasurementLayer(n_qubits=2, observable='Z'))

    qnn.build()
    qnn.compile(loss='mse')
    qnn.fit(X_train, y_train, epochs=25, verbose=False)

    # Different batch sizes
    print("\nPredicting with different batch sizes...")
    pred_full = qnn.predict(X_test, batch_size=None)
    pred_batch = qnn.predict(X_test, batch_size=5)
    pred_raw = qnn.predict(X_test, return_raw=True)

    print(f"Full batch:     {pred_full}")
    print(f"Mini-batch (5): {pred_batch}")
    print(f"True labels:    {y_test}")
    print(f"\nRaw expectation values: {pred_raw[:5]}")

    accuracy = qnn.evaluate(X_test, y_test)
    print(f"\nAccuracy: {accuracy:.4f}")

    return qnn


# ==================== EXAMPLE 7: SAVE/LOAD ====================

def example_7_save_load_model():
    """Example 7: Save and load model"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Model Save and Load")
    print("="*80)

    X_train, X_test, y_train, y_test = prepare_iris_binary_dataset()

    # Train and save
    print("\n1. Training model...")
    qnn = QuantumNeuralNetwork_Basic_CPU(n_qubits=2, name="SaveLoadQNN", shots=1024)
    qnn.add(EncodingLayer(n_qubits=2, encoding_type='angle'))
    qnn.add(VariationalLayer(n_qubits=2, n_layers=2, entanglement='linear'))
    qnn.add(MeasurementLayer(n_qubits=2, observable='Z'))

    qnn.build()
    qnn.compile(loss='mse')
    qnn.fit(X_train, y_train, epochs=20, verbose=False)

    original_acc = qnn.evaluate(X_test, y_test)
    print(f"   Original accuracy: {original_acc:.4f}")

    print("\n2. Saving model...")
    qnn.save_model("trained_qnn.pkl", save_history=True)

    # Load
    print("\n3. Loading model...")
    loaded_data = QuantumNeuralNetwork_Basic_CPU.load_model("trained_qnn.pkl")

    print(f"   Name: {loaded_data['name']}")
    print(f"   Qubits: {loaded_data['n_qubits']}")
    print(f"   Parameters: {loaded_data['parameters'].shape}")

    # Reconstruct
    print("\n4. Reconstructing model...")
    qnn_new = QuantumNeuralNetwork_Basic_CPU(
        n_qubits=loaded_data['n_qubits'],
        name=loaded_data['name'],
        shots=loaded_data['shots']
    )
    qnn_new.add(EncodingLayer(n_qubits=2, encoding_type='angle'))
    qnn_new.add(VariationalLayer(n_qubits=2, n_layers=2, entanglement='linear'))
    qnn_new.add(MeasurementLayer(n_qubits=2, observable='Z'))
    qnn_new.build()
    qnn_new.parameters = loaded_data['parameters']

    new_acc = qnn_new.evaluate(X_test, y_test)
    print(f"   Reconstructed accuracy: {new_acc:.4f}")
    print(f"   Match: {np.isclose(original_acc, new_acc)}")

    return qnn_new


# ==================== MAIN ====================

if __name__ == "__main__":
    print("="*80)
    print("QUANTUM NEURAL NETWORK FRAMEWORK - COMPLETE EXAMPLES")
    print("="*80)

    try:
        # Run examples
        qnn1, acc1 = example_1_basic_training()
        qnn2, acc2 = example_2_advanced_features()
        cv_res = example_3_cross_validation()
        enc_res = example_4_encoding_comparison()
        ent_res = example_5_entanglement_comparison()
        qnn6 = example_6_batch_prediction()
        qnn7 = example_7_save_load_model()

        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED!")
        print("="*80)
        print("\nGenerated files:")
        print("  • basic_qnn_model.pkl")
        print("  • basic_qnn_circuit.png")
        print("  • best_qnn.pkl")
        print("  • trained_qnn.pkl")
        print("  • trained_qnn_metadata.json")
        print("  • training_history.png")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
