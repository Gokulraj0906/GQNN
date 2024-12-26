from QNN.data.dataset import Data_Read

data_dir = '/home/gokulraj/Projects/Projects/Jupyter_notebook/Employee_Salary_Dataset.csv'

df = Data_Read.Read_csv(data_dir)

print("Original DataFrame (after reading and cleaning):")
print(df.head())


df_with_encoded_columns = Data_Read.convert_strings_to_numeric()

print("\nDataFrame after One-Hot Encoding of string columns:")
print(df_with_encoded_columns.head())


scaled_df = Data_Read.Scale_data(method='minmax')


print("\nScaled DataFrame (using Min-Max Scaling):")
print(scaled_df.head())


scaled_df_specific_columns = Data_Read.Scale_data(method='minmax', columns=['Salary', 'Age'])
print("\nScaled DataFrame (only 'Salary' and 'Age' columns):")
print(scaled_df_specific_columns.head())