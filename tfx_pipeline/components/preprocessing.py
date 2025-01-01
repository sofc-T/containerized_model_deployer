"""
Preprocessing module for the TFX pipeline.

This module contains functions to load, clean, and preprocess data
for training and evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """
    Clean the dataset by handling missing values and removing duplicates.
    
    Args:
        data (pd.DataFrame): Raw dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Drop rows with missing values
    data = data.dropna()
    # Remove duplicate rows
    data = data.drop_duplicates()
    return data

def preprocess_features(data, target_column):
    """
    Preprocess features and split the dataset into training and testing sets.
    
    Args:
        data (pd.DataFrame): Cleaned dataset.
        target_column (str): Name of the target column.
    
    Returns:
        tuple: Scaled features and labels for training and testing.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def preprocess_data(file_path, target_column):
    """
    End-to-end preprocessing pipeline for the dataset.
    
    Args:
        file_path (str): Path to the dataset file.
        target_column (str): Name of the target column.
    
    Returns:
        tuple: Preprocessed features and labels for training and testing.
    """
    data = load_data(file_path)
    data = clean_data(data)
    X_train, X_test, y_train, y_test = preprocess_features(data, target_column)
    return X_train, X_test, y_train, y_test
