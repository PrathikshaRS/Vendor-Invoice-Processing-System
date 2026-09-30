import sqlite3
import pandas as pd 
from sklearn.model_selection import train_test_split

def load_data_from_db(db_path, str):
    """
    Load data from a SQLite database table into a pandas DataFrame.

    Parameters:
    db_path (str): Path to the SQLite database file.
    table_name (str): Name of the table to load data from.

    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Load data into a DataFrame
    query = f"SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    
    # Close the connection
    conn.close()
    
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Select features and target variable from the DataFrame.
    """
    X = df[['Dollars']]
    y = df[['Freight']]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

