import unittest
import os
import pandas as pd
from io import StringIO
from QNN.data.dataset import Data_Read

class TestDataRead(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up mock data and files for testing."""
        # Mock CSV data
        cls.csv_data = StringIO("""A,B,C,D
1,2,3,cat
4,5,6,dog
7,8,9,cat
""")
        cls.excel_data = pd.DataFrame({
            'A': [1, 4, 7],
            'B': [2, 5, 8],
            'C': [3, 6, 9],
            'D': ['cat', 'dog', 'cat']
        })
        cls.json_data = '[{"A": 1, "B": 2, "C": 3, "D": "cat"}, {"A": 4, "B": 5, "C": 6, "D": "dog"}]'
        
        # Create test files
        cls.csv_path = 'test_data.csv'
        cls.excel_path = 'test_data.xlsx'
        cls.json_path = 'test_data.json'
        cls.sql_db = 'test_data.db'

        with open(cls.csv_path, 'w') as f:
            f.write(cls.csv_data.getvalue())
        
        cls.excel_data.to_excel(cls.excel_path, index=False)
        
        with open(cls.json_path, 'w') as f:
            f.write(cls.json_data)
        
        # Set up SQLite database
        import sqlite3
        conn = sqlite3.connect(cls.sql_db)
        cls.excel_data.to_sql('test_table', conn, index=False, if_exists='replace')
        conn.close()

    @classmethod
    def tearDownClass(cls):
        """Remove mock files after testing."""
        os.remove(cls.csv_path)
        os.remove(cls.excel_path)
        os.remove(cls.json_path)
        os.remove(cls.sql_db)

    def test_read_csv(self):
        df = Data_Read.Read_csv(self.csv_path)
        self.assertEqual(len(df), 3)  # Verify row count
        self.assertIn('D_cat', df.columns)  # Verify one-hot encoding (ensure this matches the output)

    def test_read_excel(self):
        df = Data_Read.Read_excel(self.excel_path)
        self.assertEqual(len(df), 3)  # Verify row count
        self.assertIn('D_dog', df.columns)  # Verify one-hot encoding (ensure this matches the output)

    def test_read_json(self):
        df = Data_Read.Read_json(self.json_path)
        self.assertEqual(len(df), 2)  # Verify row count
        self.assertIn('D_dog', df.columns)  # Verify one-hot encoding (ensure this matches the output)

    def test_read_sql(self):
        df = Data_Read.Read_sql(self.sql_db, "SELECT * FROM test_table")
        self.assertEqual(len(df), 3)  # Verify row count
        self.assertIn('D_dog', df.columns)  # Verify one-hot encoding (ensure this matches the output)

    def test_scale_data(self):
        df = Data_Read.Read_csv(self.csv_path)
        scaled_df = Data_Read.Scale_data(df, method='minmax', columns=['A', 'B', 'C'])
        self.assertAlmostEqual(scaled_df['A'].max(), 1.0)  # Verify scaling
        self.assertAlmostEqual(scaled_df['A'].min(), 0.0)

    def test_clean_data(self):
        dirty_data = pd.DataFrame({
            'A': [1, 1, 2],
            'B': [2, None, 3],
            'C': [None, 4, 5]
        })
        cleaned_data = Data_Read.clean_data(dirty_data)
        self.assertEqual(len(cleaned_data), 1)  # Verify duplicate and null removal
    
    def test_convert_strings_to_numeric(self):
        df = pd.DataFrame({
            'A': [1, 2],
            'B': ['cat', 'dog']
        })
        encoded_df = Data_Read.convert_strings_to_numeric(df, columns=['B'])
        self.assertIn('B_dog', encoded_df.columns)  # Verify one-hot encoding

if __name__ == '__main__':
    unittest.main()
