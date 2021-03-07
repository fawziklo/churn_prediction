import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = ['ADB', 'GBC', 'RF', 'XGB', 'SVC']

clfs = [
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(n_jobs=-1),
    XGBClassifier(),
    SVC(probability=True)
]

params = {
    models[0]: {'learning_rate': [1, 0.01], 'n_estimators': [500, 1000]},
    models[1]: {'learning_rate': [0.01], 'n_estimators': [500, 1000], 'max_depth': [3],
                'min_samples_split': [2], 'min_samples_leaf': [2]},
    models[2]: {'n_estimators': [500, 1000, 1500], 'criterion': ['gini'], 'min_samples_split': [2],
                'min_samples_leaf': [4]},
    models[3]: {},
    models[4]: {'C': [0.01, 1, 10, 100], 'gamma': [1, 0.1], 'kernel': ['rbf', 'linear']},

}


def calculate_best_clf(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
    for name, estimator in zip(models, clfs):
        print("Performing : " + name)
        clf = GridSearchCV(estimator, params[name], n_jobs=-1, cv=10)

        clf.fit(x_train, y_train)

        print("best params: " + str(clf.best_params_))
        print("best scores: " + str(clf.best_score_))
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy: {:.4%}".format(acc))

        # save the model to disk
        if acc >= .8 and clf.best_score_ >= .8:
            joblib.dump(clf, 'data_model_results/' + name + str(clf.best_score_))
