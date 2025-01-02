from GQNN.data.dataset import Data_Read
from GQNN.models.data_split import DataSplitter
from GQNN.models.Linear_model import QuantumClassifier_EstimatorQNN
import numpy as np 



data_dir = '/home/gokulraj/Projects/Projects/GQNN/QNN/examples/Employee_Salary_Dataset.csv'

df = Data_Read.Read_csv(data_dir)

print("Original DataFrame (after reading and cleaning):")
print(df.head())


df_with_encoded_columns = Data_Read.convert_strings_to_numeric()

print("\nDataFrame after One-Hot Encoding of string columns:")
print(df_with_encoded_columns.head())


scaled_df = Data_Read.Scale_data(method='minmax')


print("\nScaled DataFrame (using Min-Max Scaling):")
print(scaled_df.head())

x = df_with_encoded_columns.drop('Gender_Male',axis=1)
y = df_with_encoded_columns['Gender_Male'].astype(int)


split = DataSplitter(x,y,0.75,True,43)

x_train,x_test,y_train,y_test = split.split()

x_train = np.array(x_train)
y_train = np.array(y_train)

model = QuantumClassifier_EstimatorQNN(num_qubits=4,maxiter=60,random_seed=143)

model.fit(x_train,y_train)

model.print_model()

score = model.score(x_test,y_test)
print(f"Model accuracy: {score * 100:.2f}%")
