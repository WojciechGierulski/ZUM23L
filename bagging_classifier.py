import random

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, Any, Optional, Union, Tuple, Type, List
import copy
from tqdm import tqdm


class CustomBaggingClassifier(BaseEstimator):
    def __init__(self, estimator: Type[BaseEstimator],
                 n_estimators: int = 10,
                 sampling_with_replacement: bool = True,
                 sampling_with_replacement_features: bool = False,
                 sample_size: float = 1.0,
                 sample_size_features: float = 1.0,
                 estimator_kwargs: Optional[Dict[str, Any]] = None,
                 categorical_features: Optional[List[str]] = None
                 ):
        super().__init__()
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.n_classes = None
        self.sampling_with_replacement = sampling_with_replacement
        self.sampling_with_replacement_features = sampling_with_replacement_features
        self.sample_size = sample_size
        self.sample_size_features = sample_size_features
        if estimator_kwargs is None:
            estimator_kwargs = {}
        self.estimator_kwargs = estimator_kwargs
        self.estimators = [self.estimator(**self.estimator_kwargs) for _ in range(self.n_estimators)]
        self.estimators_parameters = [
            {"samples_indices": None, "features": None, "transformer": None}  # type: ignore
            for _ in range(self.n_estimators)]
        self.has_predict_proba = hasattr(estimator, "predict_proba")
        if categorical_features is None:
            self.categorical_features = set()
        else:
            self.categorical_features = categorical_features

    @staticmethod
    def _convert_to_pd(X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Converts np.ndarray to dataframe for future consistency
        """
        if type(X) is not pd.DataFrame:
            X_new = pd.DataFrame(data=X, columns=list(range(len(X.shape[1]))))
        else:
            X_new = copy.deepcopy(X)
        if type(y) is not pd.DataFrame:
            y_new = pd.DataFrame(data=y, columns=[0])
        else:
            y_new = copy.deepcopy(y)
        return X_new, y_new

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame],):
        X, y = self._convert_to_pd(X, y)
        self.X = X
        self.y = y
        self.n_classes = y.nunique()[0]
        for model_nr in range(self.n_estimators):
            features_sample = np.random.choice(X.columns, size=int(self.sample_size_features * X.shape[1]),
                                               replace=self.sampling_with_replacement_features)
            features_sample = list(set(features_sample))
            while True: # sample till all classes are present
                X_sample: pd.DataFrame = X.sample(frac=self.sample_size, replace=self.sampling_with_replacement)[
                    features_sample]
                y_sample: pd.DataFrame = y.loc[X_sample.index]
                if y_sample.nunique()[0] == self.n_classes:
                    break
            self.estimators_parameters[model_nr]["features"] = features_sample
            self.estimators_parameters[model_nr]["samples_indices"] = X_sample.index
            numerical_features = list(set(X_sample.columns) - set(self.categorical_features))
            curr_categorical_features = list(set(self.categorical_features).intersection(set(features_sample)))
            self.estimators_parameters[model_nr]["transformer"] = ColumnTransformer(
                [('ct', StandardScaler(), numerical_features),
                 ("ohe", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), curr_categorical_features)],
                remainder='passthrough')
            X_sample_transformed = self.estimators_parameters[model_nr]["transformer"].fit_transform(X_sample)
            self.estimators[model_nr].fit(X_sample_transformed, y_sample)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], model_nrs: Optional[List[int]]=None, predict_proba=False) -> np.ndarray:
        classes_arr = np.zeros((X.shape[0], self.n_classes))
        if model_nrs is None:
            model_nrs = list(range(self.n_estimators))
        if type(X) is np.ndarray:
            X = pd.DataFrame(data=X, columns=list(range(len(X.shape[1]))))
        for i, (estimator_params, estimator) in enumerate(zip(self.estimators_parameters, self.estimators)):
            if i not in model_nrs:
                continue
            X_sample = X[estimator_params["features"]]
            X_sample = estimator_params["transformer"].transform(X_sample)
            if self.has_predict_proba:
                y_pred = estimator.predict_proba(X_sample)
                classes_arr += y_pred
            else:
                y_pred = estimator.predict(X_sample)
                for i, x in enumerate(y_pred):
                    classes_arr[i, x] += 1
        if not predict_proba:
            return classes_arr.argmax(axis=1)
        else:
            return classes_arr / classes_arr.sum(axis=1)[:,np.newaxis]
        

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.predict(X, predict_proba=True)