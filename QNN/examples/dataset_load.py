from QNN.data.dataset import Data_Read
from QNN.models.data_split import DataSplitter
from QNN.models.Linear_model import LinearRegression as QNN_LinearRegression
from QNN.data.rfe import FeatureSelector

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




# scaled_df_specific_columns = Data_Read.Scale_data(method='minmax', columns=['Salary', 'Age'])
# print("\nScaled DataFrame (only 'Salary' and 'Age' columns):")
# print(scaled_df_specific_columns.head())

from sklearn.linear_model import LinearRegression
from QNN.data.rfe import FeatureSelector  

model = LinearRegression()

selector = FeatureSelector(estimator=model, task='regression', step=1, cv=5)

X = scaled_df.drop(columns=['Salary'], axis=1)
y = scaled_df['Salary']

X_selected = selector.fit_transform(X, y)

print("Selected Features:", selector.get_selected_features())



x = scaled_df[['Experience_Years','Age']]
y = scaled_df['Salary']

split = DataSplitter(X=x,y=y,train_size=0.7,shuffle=False,random_state=43)

x_train,x_test,y_train,y_test = split.split()

model = QNN_LinearRegression()
model.train(x_train,y_train)

y_pred = model.predict(x_test)

score = model.score(x_test,y_test)


print(score)