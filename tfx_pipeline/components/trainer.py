"""
Trainer module for the TFX pipeline.

This module defines the model architecture, trains the model, evaluates it,
and exports the trained model.
"""

import tensorflow as tf
from sklearn.metrics import mean_squared_error, accuracy_score

def create_model(input_shape, output_units, output_activation="linear"):
    """
    Define the model architecture.

    Args:
        input_shape (tuple): Shape of the input data (excluding batch size).
        output_units (int): Number of output units.
        output_activation (str): Activation function for the output layer.

    Returns:
        tf.keras.Model: Compiled TensorFlow model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(output_units, activation=output_activation)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, output_units, epochs=20, batch_size=32):
    """
    Train the model.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.
        input_shape (tuple): Shape of the input data (excluding batch size).
        output_units (int): Number of output units.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        tf.keras.Model: Trained model.
    """
    model = create_model(input_shape, output_units)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model

def evaluate_model(model, X_test, y_test, task_type="regression"):
    """
    Evaluate the trained model.

    Args:
        model (tf.keras.Model): Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test labels.
        task_type (str): Type of task ('regression' or 'classification').

    Returns:
        dict: Evaluation metrics.
    """
    predictions = model.predict(X_test)
    if task_type == "regression":
        mse = mean_squared_error(y_test, predictions)
        return {"Mean Squared Error": mse}
    elif task_type == "classification":
        predictions = tf.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_test, predictions)
        return {"Accuracy": accuracy}
    else:
        raise ValueError("Unsupported task type. Use 'regression' or 'classification'.")

def save_model(model, save_path="models/saved_model/"):
    """
    Save the trained model.

    Args:
        model (tf.keras.Model): Trained model.
        save_path (str): Path to save the model.
    """
    model.save(save_path)
    print(f"Model saved to {save_path}")

def train_pipeline(X_train, X_val, X_test, y_train, y_val, y_test, input_shape, output_units, task_type="regression"):
    """
    End-to-end training pipeline.

    Args:
        X_train (numpy.ndarray): Training features.
        X_val (numpy.ndarray): Validation features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training labels.
        y_val (numpy.ndarray): Validation labels.
        y_test (numpy.ndarray): Test labels.
        input_shape (tuple): Shape of the input data.
        output_units (int): Number of output units.
        task_type (str): Type of task ('regression' or 'classification').

    Returns:
        tf.keras.Model: Trained model.
    """
    model = train_model(X_train, y_train, X_val, y_val, input_shape, output_units)
    metrics = evaluate_model(model, X_test, y_test, task_type)
    print("Evaluation Metrics:", metrics)
    save_model(model)
    return model
