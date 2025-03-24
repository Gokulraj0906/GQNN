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
    import platform
    if platform.system().lower() == "linux":
        try:
            import fireducks.pandas as pd
            print("🚀 Linux Kernel detected! Time to unleash the power of open-source computing! 🐧")
        except ImportError:
            import pandas as pd
            print("🚀 Linux detected, but fireducks.pandas is not available. Using standard pandas.")
    else:
        import pandas as pd
        if platform.system().lower() == "darwin":
            print("🍏 macOS detected! Let's innovate with style and efficiency! 🚀")
        else:
            print("🌍 Running on Windows! Let's make some magic happen across platforms! 🎩✨")
    
    def __init__(self):
        """
        Initializes the Data_Read class with default attributes.
        """
        self.data_path = None
        self.df = None

    @staticmethod
    def __clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset by:
        - Removing duplicate rows
        - Handling missing values efficiently with forward and backward filling

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: A cleaned dataframe without duplicates and efficiently handled missing values.
        """
        df.drop_duplicates(inplace=True)
        df.dropna(how='all', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df

    @staticmethod
    def _get_file_path(data_path: str, file_extension: str) -> str:
        """ 
        Retrieves the file path, whether it's a directory or a specific file.
        
        Args:
            data_path (str): The path to the dataset.
            file_extension (str): The expected file extension.
        
        Returns:
            str: The resolved file path.
        
        Raises:
            FileNotFoundError: If no file is found or an incorrect path is given.
        """
        import os
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
        Converts categorical string columns into numeric using One-Hot Encoding.
        This transformation is applied only if explicitly requested.
        
        Args:
            columns (list, optional): List of columns to convert. Defaults to all string columns.
        
        Returns:
            pd.DataFrame: The transformed dataframe with categorical values encoded.
        
        Raises:
            ValueError: If the dataframe is empty or specified columns are not of type 'object'.
        """
        
        if cls.df is None:
            raise ValueError("No data available to convert. Please read data first.")
        
        if columns is None:
            columns = cls.df.select_dtypes(include=['object']).columns.tolist()
        
        non_string_columns = [col for col in columns if cls.df[col].dtype != 'object']
        if non_string_columns:
            raise ValueError(f"Columns {non_string_columns} are not of string type.")
        
        cls.df = cls.pd.get_dummies(cls.df, columns=columns, drop_first=True)
        return cls.df

    @classmethod
    def Read_csv(cls, data_path: str, convert_strings: bool = False) -> pd.DataFrame:
        """
        Reads a CSV file and loads it into a dataframe.
        Optionally, categorical string columns can be converted to numeric.
        
        Args:
            data_path (str): Path to the CSV file.
            convert_strings (bool, optional): Whether to convert string columns to numeric. Defaults to False.
        
        Returns:
            pd.DataFrame: The loaded and cleaned dataframe.
        """
        path = cls._get_file_path(data_path, '.csv')
        cls.data_path = path
        df = cls.pd.read_csv(path)
        cls.df = cls.__clean_data(df)
        if convert_strings:
            cls.convert_strings_to_numeric()
        return cls.df

    @classmethod
    def Read_excel(cls, data_path: str, convert_strings: bool = False) -> pd.DataFrame:
        """
        Reads an Excel file and loads it into a dataframe.
        Optionally, categorical string columns can be converted to numeric.
        
        Args:
            data_path (str): Path to the Excel file.
            convert_strings (bool, optional): Whether to convert string columns to numeric. Defaults to False.
        
        Returns:
            pd.DataFrame: The loaded and cleaned dataframe.
        """
        path = cls._get_file_path(data_path, '.xlsx')
        cls.data_path = path
        df = cls.pd.read_excel(path)
        cls.df = cls.__clean_data(df)
        if convert_strings:
            cls.convert_strings_to_numeric()
        return cls.df

    @classmethod
    def Read_json(cls, data_path: str) -> pd.DataFrame:
        """
        Reads a JSON file and loads it into a dataframe.
        
        Args:
            data_path (str): Path to the JSON file.
        
        Returns:
            pd.DataFrame: The loaded and cleaned dataframe.
        """
        path = cls._get_file_path(data_path, '.json')
        cls.data_path = path
        df = cls.pd.read_json(path)
        cls.df = cls.__clean_data(df)
        return cls.df

    @classmethod
    def Read_sql(cls, query: str, connection) -> pd.DataFrame:
        """
        Reads data from an SQL database using a query.
        
        Args:
            query (str): The SQL query to execute.
            connection (str or sqlite3.Connection): A database connection string or connection object.
        
        Returns:
            pd.DataFrame: The loaded and cleaned dataframe.
        """
        import sqlite3
        if isinstance(connection, str):
            conn = sqlite3.connect(connection)
        else:
            conn = connection
        df = cls.pd.read_sql_query(query, conn)
        cls.df = cls.__clean_data(df)
        return cls.df

    @classmethod
    def Scale_data(cls, method: str = 'minmax', columns: list = None) -> pd.DataFrame:
        """
        Scales numeric data using different scaling methods.
        
        Available scaling methods:
            - 'minmax': Min-Max Scaling (values between 0 and 1)
            - 'zscale': Standard Scaling (mean = 0, std = 1)
            - 'robust': Robust Scaling (handles outliers)
        
        Args:
            method (str, optional): The scaling method to use. Defaults to 'minmax'.
            columns (list, optional): List of numeric columns to scale. Defaults to all numeric columns.
        
        Returns:
            pd.DataFrame: The dataframe with scaled values.
        
        Raises:
            ValueError: If the dataframe is empty or specified columns are non-numeric.
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