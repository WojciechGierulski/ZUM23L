import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from bagging_classifier import CustomBaggingClassifier
from datasets import load_dataset
from tqdm import tqdm


def get_k_fold_metrics(model, splits, X, y):
    cv = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
    micro_f1_all = []
    macro_f1_all = []
    micro_auc_all = []
    macro_auc_all = []
    for fold, (train, test) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        if y_pred_proba.shape[1] == 2: # binary classification
            y_pred_proba = y_pred_proba[:,1]

        micro_f1 = f1_score(y_test.values.squeeze(1), y_pred, average='micro')
        macro_f1 = f1_score(y_test.values.squeeze(1), y_pred, average='macro')
        micro_auc = roc_auc_score(y_test.values.squeeze(1), y_pred_proba, multi_class='ovr', average='micro')
        macro_auc = roc_auc_score(y_test.values.squeeze(1), y_pred_proba, multi_class='ovr', average='macro')
        micro_f1_all.append(micro_f1)
        macro_f1_all.append(macro_f1)
        macro_auc_all.append(macro_auc)
        micro_auc_all.append(micro_auc)

    total_micro_auc = sum(micro_auc_all) / splits
    total_macro_auc = sum(macro_auc_all) / splits
    total_micro_f1 = sum(micro_f1_all) / splits
    total_macro_f1 = sum(macro_f1_all) / splits
    return pd.DataFrame.from_dict(data=[{'total_micro_auc': total_micro_auc,
                         'total_macro_auc': total_macro_auc,
                         'total_micro_f1': total_micro_f1,
                         'total_macro_f1': total_macro_f1,
                         'sum': total_micro_auc + total_macro_auc + total_micro_f1 + total_macro_f1}])

def get_train_test_metrics(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)

    micro_f1 = f1_score(y_test.values.squeeze(1), y_pred, average='micro')
    macro_f1 = f1_score(y_test.values.squeeze(1), y_pred, average='macro')
    micro_auc = roc_auc_score(y_test.values.squeeze(1), y_pred_proba, multi_class='ovr', average='micro')
    macro_auc = roc_auc_score(y_test.values.squeeze(1), y_pred_proba, multi_class='ovr', average='macro')

    return pd.DataFrame.from_dict(data=[{'total_micro_auc': micro_auc,
                         'total_macro_auc': macro_auc,
                         'total_micro_f1': micro_f1,
                         'total_macro_f1': macro_f1,
                         'sum': micro_auc + macro_auc + micro_f1 + macro_f1}])


def fast_grid_search(grid, base_model, categorical_columns, X, y, splits=5):
    sklearn_grid = ParameterGrid(grid)
    total_metrics = pd.DataFrame()

    for param_grid in tqdm(sklearn_grid):
        model = CustomBaggingClassifier(estimator=base_model,
                                        sample_size=0.8,
                                        sample_size_features=0.8,
                                        sampling_with_replacement_features=False,
                                        sampling_with_replacement=False,
                                        n_estimators=5,
                                        categorical_features=categorical_columns,
                                        estimator_kwargs=param_grid)
        metrics = get_k_fold_metrics(model, splits=splits, X=X, y=y)
        row = pd.concat([pd.DataFrame([param_grid]), metrics], axis=1)
        total_metrics = pd.concat([total_metrics, row])
    return total_metrics


def n_estimators_effect(base_model, n_estimators_arr, categorical_columns, X_train, X_test, y_train, y_test, estimator_kwargs, title):
    total_metrics = pd.DataFrame()
    for n_estimators in tqdm(n_estimators_arr):
        model = CustomBaggingClassifier(estimator=base_model,
                                sample_size=0.8,
                                sample_size_features=0.8,
                                sampling_with_replacement_features=False,
                                sampling_with_replacement=False,
                                n_estimators=n_estimators,
                                categorical_features=categorical_columns,
                                estimator_kwargs=estimator_kwargs)
        metrics = get_train_test_metrics(model, X_train, X_test, y_train, y_test)
        total_metrics = pd.concat([total_metrics, metrics])

    plt.figure()
    for metric in ['total_micro_auc','total_macro_auc','total_micro_f1','total_macro_f1']:
        plt.plot(n_estimators_arr, total_metrics[metric], '-o', label=metric)
    plt.legend(loc='best')
    plt.xlabel("n_estimators")
    plt.title(title)
    plt.ylim([0,1])


def sample_size_effect(base_model, sample_size_arr, categorical_columns, X_train, X_test, y_train, y_test, estimator_kwargs, title, n_estimators):
    total_metrics = pd.DataFrame()
    for sample_size in tqdm(sample_size_arr):
        model = CustomBaggingClassifier(estimator=base_model,
                                sample_size=sample_size,
                                sample_size_features=0.8,
                                sampling_with_replacement_features=False,
                                sampling_with_replacement=False,
                                n_estimators=n_estimators,
                                categorical_features=categorical_columns,
                                estimator_kwargs=estimator_kwargs)
        metrics = get_train_test_metrics(model, X_train, X_test, y_train, y_test)
        total_metrics = pd.concat([total_metrics, metrics])

    plt.figure()
    for metric in ['total_micro_auc','total_macro_auc','total_micro_f1','total_macro_f1']:
        plt.plot(sample_size_arr, total_metrics[metric], '-o', label=metric)
    plt.legend(loc='best')
    plt.xlabel("sample_size")
    plt.title(title)
    plt.ylim([0,1])


def features_size_effect(base_model, sample_size_features_arr, categorical_columns, X_train, X_test, y_train, y_test, estimator_kwargs, title, n_estimators, sample_size):
    total_metrics = pd.DataFrame()
    for sample_size_features in tqdm(sample_size_features_arr):
        model = CustomBaggingClassifier(estimator=base_model,
                                sample_size=sample_size,
                                sample_size_features=sample_size_features,
                                sampling_with_replacement_features=False,
                                sampling_with_replacement=False,
                                n_estimators=n_estimators,
                                categorical_features=categorical_columns,
                                estimator_kwargs=estimator_kwargs)
        metrics = get_train_test_metrics(model, X_train, X_test, y_train, y_test)
        total_metrics = pd.concat([total_metrics, metrics])

    plt.figure()
    for metric in ['total_micro_auc','total_macro_auc','total_micro_f1','total_macro_f1']:
        plt.plot(sample_size_features_arr, total_metrics[metric], '-o', label=metric)
    plt.legend(loc='best')
    plt.xlabel("sample_size_features")
    plt.title(title)
    plt.ylim([0,1])


def replacement_effect(base_model, replacements_arr, categorical_columns, X_train, X_test, y_train, y_test, estimator_kwargs, title, n_estimators, sample_size, feature_sample_size):
    total_metrics = pd.DataFrame()
    for replacements in tqdm(replacements_arr):
        model = CustomBaggingClassifier(estimator=base_model,
                                sample_size=sample_size,
                                sample_size_features=feature_sample_size,
                                sampling_with_replacement_features=replacements['features'],
                                sampling_with_replacement=replacements['samples'],
                                n_estimators=n_estimators,
                                categorical_features=categorical_columns,
                                estimator_kwargs=estimator_kwargs)
        metrics = get_train_test_metrics(model, X_train, X_test, y_train, y_test)
        total_metrics = pd.concat([total_metrics, metrics])

    plt.figure()
    for i, metric in enumerate(['total_micro_auc','total_macro_auc','total_micro_f1','total_macro_f1']):
        plt.bar(np.array(range(len(replacements_arr))) + i*0.1, total_metrics[metric], label=metric, width=0.1)
    plt.legend(loc='best')
    plt.xlabel("sampling with replacement")
    plt.xticks(np.array(range(len(replacements_arr)))+0.15, [str(replacements) for replacements in replacements_arr], size='small', rotation=45)
    plt.title("Losowanie próbek i atrybutów ze zwracaniem")
    plt.ylim([0,1.3])
    return total_metrics


