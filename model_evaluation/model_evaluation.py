import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix, classification_report, roc_curve


def best_model_evaluation(x_test: np.array, y_test: np.array):
    # load best classifiers
    XGB = joblib.load("data_model_results/XGB0.879803984063745")
    y_pred_xg = XGB.predict(x_test)
    print(classification_report(y_test, y_pred_xg))

    RFC = joblib.load("data_model_results/RFC0.874986454183267")
    y_pred_rf = RFC.predict(x_test)

    fp_xg, tp_xg, _ = roc_curve(y_test, y_pred_xg)
    fp_rf, tp_rf, _ = roc_curve(y_test, y_pred_rf)

    plot_confusion_matrix(XGB, x_test, y_test, cmap=plt.cm.Blues)
    plot_roc_curves(fp_xg, tp_xg, fp_rf, tp_rf, )
    plt.show()


def plot_roc_curves(fp_xg, tp_xg, fp_rf, tp_rf):
    plt.figure()
    plt.plot(fp_xg, tp_xg, label="XGB")
    plt.plot(fp_rf, tp_rf, label="RFC")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title('Receiver Operating Characteristic')
