import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option('expand_frame_repr', False)


def read_data(path: str):
    return pd.read_csv(path)


def display_data_info(data: pd.DataFrame):
    print(data.head(5))
    print("\n")
    print(data.info())
    print("\n")
    print(data.describe())


def encode_to_numerical_data(raw_data: pd.DataFrame):
    for i in raw_data:
        if raw_data[i].dtype == 'object':
            raw_data[i] = factorization(raw_data, i)
    return raw_data


def factorization(raw_data: pd.DataFrame, col: str):
    return pd.factorize(raw_data[col])[0]


def extract_csv_files(raw_data: pd.DataFrame):
    phone = raw_data.query('PhoneService == "Yes"')
    internet = raw_data.query('InternetService != "No"')

    # PhoneIncome = 688392
    phone['TotalCharges'].to_csv("data/phone.csv", index=False)
    # InternetIncome = 703620
    internet['TotalCharges'].to_csv("data/internet.csv", index=False)
