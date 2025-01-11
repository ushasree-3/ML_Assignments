from typing import Union
import pandas as pd
import numpy as np
from dataclasses import dataclass



def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    correct_prediction = (y_hat == y).sum()
    total_predictions  = y.size
    accuracy = (correct_prediction)/(total_predictions)
    return accuracy


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    True_positive = ((y_hat == cls)&(y == cls)).sum()
    False_positive = ((y_hat != cls)&(y == cls)).sum()

    precision = True_positive/ (False_positive+True_positive)
    return precision


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    True_positive = ((y_hat == cls)&(y == cls)).sum()
    False_negative = ((y_hat == cls)&(y != cls)).sum()
    recall = True_positive/(False_negative+True_positive)
    return recall
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    mse = ((y_hat - y)**2).mean()
    rmse = np.sqrt(mse)
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    mae = (np.abs(y_hat - y)).mean()
    return mae

# np.random.seed(42)  # Set the seed for reproducibility

# @dataclass
# class TestData:
#     y_hat: pd.Series
#     y: pd.Series

# # Create some sample data
# y_hat_sample = pd.Series(np.random.randint(0, 2, size=100))  # Random binary predictions
# y_sample = pd.Series(np.random.randint(0, 2, size=100))  # Random binary true labels

# test_data = TestData(y_hat=y_hat_sample, y=y_sample)

# # Test the accuracy function
# accuracy_result = accuracy(test_data.y_hat, test_data.y)
# print(f"Accuracy: {accuracy_result}")

# # Test the precision function
# precision_result = precision(test_data.y_hat, test_data.y, cls=1)
# print(f"Precision: {precision_result}")

# # Test the recall function
# recall_result = recall(test_data.y_hat, test_data.y, cls=1)
# print(f"Recall: {recall_result}")

# # Test the rmse function
# rmse_result = rmse(test_data.y_hat, test_data.y)
# print(f"RMSE: {rmse_result}")

# # Test the mae function
# mae_result = mae(test_data.y_hat, test_data.y)
# print(f"MAE: {mae_result}")