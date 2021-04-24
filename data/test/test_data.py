import datatest as dt
import pandas as pd
import pytest


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def data():
    return pd.read_csv('../customer_churn_data.csv')


@pytest.mark.mandatory
def test_columns_names(data):
    real_column_names = {'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                         'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'}

    dt.validate(data.columns, real_column_names)


if __name__ == '__main__':
    import sys

    sys.exit(pytest.main(sys.argv))
