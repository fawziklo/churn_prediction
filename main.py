import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from data_ml_models.grid_search_models import calculate_best_clf
from data_partition.data_partition import create_xmatrix_ylabels, create_train_test_sets
from data_pre_processing.prep_process_data import read_data, display_data_info
from data_vizualisation.data_vizualisation import bar_plots, heatmap_plot, count_plots
from model_evaluation.model_evaluation import best_model_evaluation

if __name__ == '__main__':
    # Read data
    raw_data = read_data('data/customer_churn_data.csv')
    display_data_info(raw_data)

    # Data Visualization
    # Bar Plots
    bar_plots(raw_data)
    # Heat Map Plot
    heatmap_plot(raw_data)
    # count plot
    count_plots(raw_data)

    # to display figs one by one
    plt.show()

    # Data partition
    # Create X_Matrix and Y_labels
    x_matrix, y_labels = create_xmatrix_ylabels(raw_data)

    # Data Standardization
    x_matrix = StandardScaler().fit_transform(x_matrix)

    # UnderResampling (SMOTE tested but not efficient)
    churn_args = np.argwhere(y_labels[:] == 1)
    notChurn_args = np.argwhere(y_labels[:] == 0)

    x_reduced = np.vstack((x_matrix[0:len(churn_args)], np.squeeze(x_matrix[churn_args])))
    y_reduced = np.vstack(((y_labels[0:len(churn_args)]).reshape(1869, 1), y_labels[churn_args]))

    # Create Train and Test datasets
    X_train, X_test, y_train, y_test = create_train_test_sets(x_reduced, np.squeeze(y_reduced))

    # Model selection using gridSearch
    calculate_best_clf(X_train, y_train, X_test, y_test)

    # Best model evaluation using confusion matrix
    best_model_evaluation(X_test, y_test)
