"""
Comprehensive QSVM Testing: Classification and Regression
Demonstrates QSVC_CPU and QSVR_CPU on synthetic datasets
"""

from GQNN.models.qsvm import QSVC_CPU, QSVR_CPU
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_qsvc():
    """Test Quantum Support Vector Classifier"""
    print_header("QUANTUM SUPPORT VECTOR CLASSIFIER (QSVC)")
    
    # Generate classification data
    print("\n📊 Generating classification dataset...")
    X, y = make_classification(
        n_samples=80,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Classes: {np.unique(y)}")
    
    # Initialize and train QSVC
    print("\n🔬 Initializing QSVC...")
    qsvc_model = QSVC_CPU(num_qubits=2, feature_map_reps=2)
    qsvc_model.fit(X_train_scaled, y_train, verbose=True)
    
    # Make predictions
    y_pred_train = qsvc_model.predict(X_train_scaled)
    y_pred_test = qsvc_model.predict(X_test_scaled)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Display results
    print("\n" + "-"*70)
    print("📈 QSVC Performance Metrics")
    print("-"*70)
    print(f"Training Accuracy:    {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:        {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    print(f"Training Time:        {qsvc_model.training_time:.2f} seconds")
    print("-"*70)
    print("\n📊 Confusion Matrix:")
    print(cm)
    print("-"*70)

    model_circuit = qsvc_model.print_model("qsvc_circuit.png")
    print("✓ Model circuit saved as 'qsvc_circuit.png'")
    saved_model = qsvc_model.save_model("qsvc_model.pkl")
    print("✓ Model saved as 'qsvc_model.pkl'")

    
    return {
        'model': qsvc_model,
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': qsvc_model.training_time,
        'model_circuit': model_circuit,
        'saved_model': saved_model
    }


def test_qsvr():
    """Test Quantum Support Vector Regressor"""
    print_header("QUANTUM SUPPORT VECTOR REGRESSOR (QSVR)")
    
    # Generate regression data
    print("\n📊 Generating regression dataset...")
    X, y = make_regression(
        n_samples=80,
        n_features=2,
        n_informative=2,
        n_targets=1,
        noise=10.0,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Initialize and train QSVR
    print("\n🔬 Initializing QSVR...")
    qsvr_model = QSVR_CPU(
        num_qubits=2,
        feature_map_reps=2,
        epsilon=0.1
    )
    qsvr_model.fit(X_train_scaled, y_train, verbose=True)
    
    # Make predictions
    y_pred_train = qsvr_model.predict(X_train_scaled)
    y_pred_test = qsvr_model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    
    # Display results
    print("\n" + "-"*70)
    print("📈 QSVR Performance Metrics")
    print("-"*70)
    print(f"Training R² Score:    {train_r2:.4f}")
    print(f"Test R² Score:        {test_r2:.4f}")
    print(f"RMSE:                 {rmse:.4f}")
    print(f"MAE:                  {mae:.4f}")
    print(f"MSE:                  {mse:.4f}")
    print(f"Training Time:        {qsvr_model.training_time:.2f} seconds")
    print("-"*70)
    
    # Show sample predictions
    print("\n📊 Sample Predictions (first 5 test samples):")
    print("-"*70)
    print(f"{'Actual':<20} {'Predicted':<20} {'Error':<20}")
    print("-"*70)
    for i in range(min(5, len(y_test))):
        error = abs(y_test[i] - y_pred_test[i])
        print(f"{y_test[i]:<20.4f} {y_pred_test[i]:<20.4f} {error:<20.4f}")
    print("-"*70)

    model_circuit = qsvr_model.print_model("qsvr_circuit.png")
    print("✓ Model circuit saved as 'qsvr_circuit.png'")
    saved_model = qsvr_model.save_model("qsvr_model.pkl")
    print("✓ Model saved as 'qsvr_model.pkl'")

    
    return {
        'model': qsvr_model,
        'r2_score': test_r2,
        'rmse': rmse,
        'mae': mae,
        'training_time': qsvr_model.training_time,
        'model_circuit': model_circuit,
        'saved_model': saved_model
    }


def main():
    """Main execution function"""
    print_header("🌟 QUANTUM SUPPORT VECTOR MACHINE (QSVM) COMPREHENSIVE TEST 🌟")
    
    # Test Classification
    qsvc_results = test_qsvc()
    
    # Test Regression
    qsvr_results = test_qsvr()
    
    # Summary comparison
    print_header("📊 SUMMARY COMPARISON")
    print("\n" + "-"*70)
    print("Model Type       | Primary Metric      | Training Time")
    print("-"*70)
    print(f"QSVC (Classify)  | Accuracy: {qsvc_results['accuracy']:.4f}   | "
          f"{qsvc_results['training_time']:.2f}s")
    print(f"QSVR (Regress)   | R² Score: {qsvr_results['r2_score']:.4f}   | "
          f"{qsvr_results['training_time']:.2f}s")
    print("-"*70)
    
    return qsvc_results, qsvr_results


if __name__ == "__main__":
    qsvc_results, qsvr_results = main()