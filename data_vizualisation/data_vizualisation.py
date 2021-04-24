import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data_pre_processing.prep_process_data import encode_to_numerical_data


def bar_plots(raw_data: pd.DataFrame):
    fig, axes = plt.subplots(4, 2, sharex=True, figsize=(20, 10))
    fig.suptitle('Summary')
    sns.barplot(ax=axes[0, 0], x="tenure", y="Contract", hue="gender", data=raw_data)
    sns.barplot(ax=axes[0, 1], x="tenure", y="Contract", hue="PaymentMethod", data=raw_data)
    sns.barplot(ax=axes[1, 0], x="tenure", y="StreamingMovies", hue="gender", data=raw_data)
    sns.barplot(ax=axes[1, 1], x="tenure", y="StreamingMovies", hue="Partner", data=raw_data)
    sns.barplot(ax=axes[2, 0], x="MonthlyCharges", y="InternetService", hue="StreamingTV", data=raw_data)
    sns.barplot(ax=axes[2, 1], x="tenure", y="OnlineSecurity", hue="DeviceProtection", data=raw_data)
    sns.barplot(ax=axes[3, 0], x="tenure", y="OnlineSecurity", hue="InternetService", data=raw_data)
    sns.barplot(ax=axes[3, 1], x="tenure", y="Contract", hue="PaperlessBilling", data=raw_data)
    plt.show()


def heatmap_plot(raw_data: pd.DataFrame):
    encoded_data = encode_to_numerical_data(raw_data)
    x_matrix = encoded_data.drop(['Churn', 'customerID'], axis=1)
    corr_map = x_matrix.corr()
    sns.heatmap(corr_map, vmax=.8, square=True, annot=True, fmt='.2f', cmap="summer")
    plt.show()


def count_plots(raw_data: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
    fig.suptitle('Summary')

    sns.countplot(ax=axes[0], x="Churn", hue="InternetService", data=raw_data)
    sns.countplot(ax=axes[1], x="Churn", hue="PhoneService", data=raw_data)
