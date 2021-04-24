import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from data_pre_processing.prep_process_data import encode_to_numerical_data


def create_xmatrix_ylabels(raw_data: pd.DataFrame):
    encoded_data = encode_to_numerical_data(raw_data)

    x_matrix = encoded_data.drop(['Churn', 'customerID'], axis=1).values
    y_labels = encoded_data['Churn'].values

    print(np.shape(x_matrix))
    print(np.shape(y_labels))

    return x_matrix, y_labels


def create_train_test_sets(x_matrix: np.array, y_labels: np.array):
    return train_test_split(x_matrix, y_labels, test_size=0.33, random_state=42)
