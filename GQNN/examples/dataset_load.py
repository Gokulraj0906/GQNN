from GQNN.data.dataset import Data_Read
from GQNN.models.data_split import DataSplitter
from GQNN.models.classification_model import QuantumClassifier_EstimatorQNN_CPU, QuantumClassifier_SamplerQNN_CPU,VariationalQuantumClassifier_CPU
import numpy as np
from GQNN.models.regression_model import QuantumRegressor_EstimatorQNN_CPU,QuantumRegressor_VQR_CPU
from joblib import dump
from sklearn.datasets import load_diabetes
# Path to dataset
data_dir = 'D:\\Projects\\GQNN\\GQNN\\examples\\Employee_Salary_Dataset.csv'

# Step 1: Load and preprocess data
df = Data_Read.Read_csv(data_dir)  # Read the CSV file
df_with_encoded_columns = Data_Read.convert_strings_to_numeric()  # Convert categorical strings to numeric
scaled_df = Data_Read.Scale_data(method='minmax')  # Scale data using Min-Max Scaling

print("\nScaled DataFrame (using Min-Max Scaling):")
print(scaled_df.head())

# Step 2: Define features (X) and target (y)
x = scaled_df.drop('Gender_Male', axis=1)  # Drop target column from features
y = scaled_df['Gender_Male'].astype(int)  # Convert target to integer

# Step 3: Split data into training and testing sets
split = DataSplitter(x, y, train_size=0.75, shuffle=True, random_state=43)
x_train, x_test, y_train, y_test = split.split()

x_train = np.array(x_train)  
y_train = np.array(y_train)  
x_test = np.array(x_test)    
y_test = np.array(y_test)    

num_qubits = x_train.shape[1]
print(f"x_train shape: {x_train.shape}")
print(f"x_train shape: {x_train.shape}")  
print(f"y_train shape: {y_train.shape}")  

# Initialize and train the Quantum Neural Network model
model = VariationalQuantumClassifier_CPU(num_qubits=4)
model.fit(x_train, y_train)

# Print the trained model's parameters
# model.print_model()

# Evaluate the model and compute accuracy
score = model.evaluate(x_test, y_test)
# adjusted_score = 1 - score
print(f"Model accuracy (adjusted): {score}%")
model.visualize_circuit()

model.save_model('VQC_model.model')
# model_1 = QuantumClassifier_SamplerQNN_CPU(num_inputs=4, output_shape=2, ansatz_reps=1,maxiter=35)
# model_1.fit(x_train, y_train)

# # Print the trained model's parameters
# model_1.print_model()

# model_1_score = model_1.score(x_test, y_test)
# print(f"Model accuracy (QuantumClassifier_SamplerQNN_CPU): {model_1_score * 100:.2f}%")

# model_2 = VariationalQuantumClassifier_CPU(num_inputs=num_qubits,max_iter=40)

# model_2.fit(x_train, y_train)

# # Print the trained model's parameters
# model_2.print_model()

# model_2_score = model_2.score(x_test, y_test)

# print(f"Model accuracy (VariationalQuantumClassifier_CPU): {model_2_score * 100:.2f}%")

# regression_model = QuantumRegressor_EstimatorQNN_CPU(num_qubits=num_qubits,maxiter=40)
# regression_model.fit(x_train, y_train)

# # Print the trained model's parameters
# regression_model.print_model()
# # dump(regression_model, 'EstimatorQNN_model.model')

# model_2_score = regression_model.score(x_test, y_test)

# print(f"Model accuracy : {model_2_score * 100:.2f}%")