import datatest as dt
import pandas as pd
import pytest

from data_partition.data_partition import create_xmatrix_ylabels, create_train_test_sets


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def data():
    return pd.read_csv('../data/customer_churn_data.csv')


@pytest.mark.mandatory
def test_columns_names(data):
    real_column_names = {'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                         'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'}

    dt.validate(data.columns, real_column_names)


@pytest.fixture(scope='module')
def test_xmatrix_ylabels(data):
    result_xmatrix, result_xlabel = create_xmatrix_ylabels(data)

    assert len(result_xmatrix) == 7043
    assert len(result_xlabel) == 7043


def test_create_train_test_sets():
    # Generate dummies data to avoid importing numpy and other external API
    xmatrix = [[1, 2, 3], [1, 2, 3],
               [1, 2, 3], [1, 2, 3],
               [1, 2, 3], [1, 2, 3],
               [1, 2, 3], [1, 2, 3],
               [1, 2, 3], [1, 2, 3]]

    ylabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    xtrain, xtest, ytrain, ytest = create_train_test_sets(xmatrix, ylabels)
    assert len(xtrain) != 0
    assert len(xtest) != 0
    assert len(ytrain) != 0
    assert len(ytest) != 0


if __name__ == '__main__':
    import sys

    sys.exit(pytest.main(sys.argv))
