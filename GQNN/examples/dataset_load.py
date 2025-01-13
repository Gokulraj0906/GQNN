from GQNN.data.dataset import Data_Read
from GQNN.models.data_split import DataSplitter
from GQNN.models.Linear_model import QuantumClassifier_EstimatorQNN_CPU
import numpy as np
from joblib import dump, load

# Data loading and preparation
data_dir = '/home/gokulraj/Projects/GQNN/GQNN/examples/Employee_Salary_Dataset.csv'

# Read and process the data
df = Data_Read.Read_csv(data_dir)

# Clean the data (convert strings to numeric, scale the data)
df_with_encoded_columns = Data_Read.convert_strings_to_numeric()
scaled_df = Data_Read.Scale_data(method='minmax')

print("\nScaled DataFrame (using Min-Max Scaling):")
print(scaled_df.head())


x = df_with_encoded_columns.drop('Gender_Male', axis=1)
y = df_with_encoded_columns['Gender_Male'].astype(int)


split = DataSplitter(x, y, 0.75, True, 43)
x_train, x_test, y_train, y_test = split.split()


x_train = np.array(x_train)
y_train = np.array(y_train)


num_qubits = x_train.shape[1]
print(f"x_train shape: {x_train.shape}")

model_1 = QuantumClassifier_EstimatorQNN_CPU(
    num_qubits=num_qubits, 
    maxiter=20,
    random_seed=143
)

model_1.fit(x_train, y_train)

model_1.print_model()

score_1 = model_1.score(x_test, y_test)
print(f"Model accuracy (on training): {score_1 * 100:.2f}%")
