from GQNN.data.dataset import Data_Read
from GQNN.models.data_split import DataSplitter
import numpy as np

data_dir = 'D:\\Projects\\GQNN\\GQNN\\examples\\Employee_Salary_Dataset.csv'


df = Data_Read.Read_csv(data_dir)  
df_with_encoded_columns = Data_Read.convert_strings_to_numeric() 
scaled_df = Data_Read.Scale_data(method='minmax')  

print("\nScaled DataFrame (using Min-Max Scaling):")
print(scaled_df.head())

x = scaled_df.drop('Gender_Male', axis=1)  
y = scaled_df['Gender_Male'].astype(int)  


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
