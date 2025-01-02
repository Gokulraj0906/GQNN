import os
import platform

if platform.system().lower() == "linux":
    import fireducks.pandas as pd
else:
    import pandas as pd

class Data_Read:
    """
    A utility class for reading, cleaning, manipulating, and scaling datasets.
    It supports reading files from local directories in CSV, Excel, JSON, and SQL formats,
    and provides methods for cleaning data, converting string columns to numeric, 
    and scaling numeric data.

    Attributes:
        data_path (str): Path to the dataset file.
        df (pd.DataFrame): The DataFrame containing the dataset.
    """
    def __init__(self):
        self.data_path = None
        self.df = None

    if 'fireducks.pandas' in pd.__name__:
        print("🚀 Linux Kernel detected! Time to unleash the power of open-source computing! 🐧")
    else:
        print("🌍 Running on Windows or MacOS! Let's make some magic happen across platforms! 🎩✨")

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset by removing duplicate rows and rows with null values.

        Args:
            df (pd.DataFrame): The input DataFrame to be cleaned.

        Returns:
            pd.DataFrame: A cleaned DataFrame with duplicates and null values removed.
        """
        df = df.drop_duplicates()
        df = df.dropna()
        return df

    @staticmethod
    def _get_file_path(data_path: str, file_extension: str) -> str:
        """
        Finds the first file with the specified extension in the directory or verifies the file path.

        Args:
            data_path (str): Directory path or file path.
            file_extension (str): The file extension to look for (e.g., '.csv').

        Returns:
            str: Full file path of the first file matching the extension.

        Raises:
            FileNotFoundError: If no matching file is found in the directory or if the path is invalid.
        """
        if os.path.isdir(data_path):
            files = [f for f in os.listdir(data_path) if f.endswith(file_extension)]
            if files:
                path = os.path.join(data_path, files[0])
                print(f"Using the file: {path}")
                return path
            else:
                raise FileNotFoundError(f"No {file_extension} file found in the directory: {data_path}")
        elif os.path.exists(data_path):
            return data_path
        else:
            raise FileNotFoundError(f"Check the File Path for '{data_path}'")

    @classmethod
    def convert_strings_to_numeric(cls, columns: list = None) -> pd.DataFrame:
        """
        Converts string-type columns into numeric features using One-Hot Encoding.

        Args:
            columns (list, optional): List of column names to convert. If None, all string-type columns are converted.

        Returns:
            pd.DataFrame: The updated DataFrame with string columns converted to numeric.

        Raises:
            ValueError: If no data is loaded or if non-string columns are included in the specified list.
        """
        if cls.df is None:
            raise ValueError("No data available to convert. Please read data first.")

        if columns is None:
            columns = cls.df.select_dtypes(include=['object']).columns.tolist()

        non_string_columns = [col for col in columns if cls.df[col].dtype != 'object']
        if non_string_columns:
            raise ValueError(f"Columns {non_string_columns} are not of string type.")

        cls.df = pd.get_dummies(cls.df, columns=columns, drop_first=True)

        return cls.df

    @classmethod
    def Read_csv(cls, data_path: str) -> pd.DataFrame:
        """
        Reads a CSV file from a directory or file path and returns a cleaned DataFrame.

        Args:
            data_path (str): Directory path or file path for the CSV file.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        path = cls._get_file_path(data_path, '.csv')
        cls.data_path = path 
        df = pd.read_csv(path)
        cls.df = cls.clean_data(df)
        cls.convert_strings_to_numeric()  
        return cls.df

    @classmethod
    def Read_excel(cls, data_path: str) -> pd.DataFrame:
        """
        Reads an Excel file from a directory or file path and returns a cleaned DataFrame.

        Args:
            data_path (str): Directory path or file path for the Excel file.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        path = cls._get_file_path(data_path, '.xlsx')
        cls.data_path = path 
        df = pd.read_excel(path)
        cls.df = cls.clean_data(df)
        cls.convert_strings_to_numeric()  
        return cls.df

    @classmethod
    def Read_json(cls, data_path: str) -> pd.DataFrame:
        """
        Reads a JSON file from a directory or file path and returns a cleaned DataFrame.

        Args:
            data_path (str): Directory path or file path for the JSON file.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        path = cls._get_file_path(data_path, '.json')
        cls.data_path = path  
        df = pd.read_json(path)
        cls.df = cls.clean_data(df)
        cls.convert_strings_to_numeric()  
        return cls.df

    @classmethod
    def Read_sql(cls, data_path: str, query: str) -> pd.DataFrame:
        """
        Reads data from a SQL database using a query and returns a cleaned DataFrame.

        Args:
            data_path (str): Path to the SQLite database file.
            query (str): SQL query to execute.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        import sqlite3
        conn = sqlite3.connect(data_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        cls.df = cls.clean_data(df)
        cls.convert_strings_to_numeric()  
        return cls.df
    
    @classmethod
    def Scale_data(cls, method: str = 'minmax', columns: list = None) -> pd.DataFrame:
        """
        Scales numeric data using the specified scaling method.

        Args:
            method (str, optional): Scaling method ('minmax', 'zscale', 'robust'). Default is 'minmax'.
            columns (list, optional): List of column names to scale. If None, all numeric columns are scaled.

        Returns:
            pd.DataFrame: DataFrame with scaled columns.

        Raises:
            ValueError: If no data is loaded or if non-numeric columns are included in the specified list.
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        if cls.df is None:
            raise ValueError("No data available to scale. Please read data first.")
        
        if columns is None:
            columns = cls.df.select_dtypes(include=['number']).columns.tolist()

        non_numeric_columns = [col for col in columns if cls.df[col].dtype not in ['float64', 'int64']]
        if non_numeric_columns:
            raise ValueError(f"Columns {non_numeric_columns} are non-numeric and cannot be scaled.")

        data_to_scale = cls.df[columns]

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscale':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaling method. Choose from 'minmax', 'zscale', 'robust'.")

        scaled_data = scaler.fit_transform(data_to_scale)
        scaled_df = cls.df.copy()
        scaled_df[columns] = scaled_data

        return scaled_df
