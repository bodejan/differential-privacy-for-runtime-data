from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

class SSModel(BaseEstimator, RegressorMixin):
    """
    Scaleout-Speedup model.
    """

    def __init__(self, instance_count_index=0, regressor=None):
        """
        Initialize the SSModel.

        Args:
            instance_count_index (int): Index within features for the instance count.
            regressor: The regressor to use for modeling.
        """
        self.regressor = regressor
        self.scales = []
        self.instance_count_index = instance_count_index

    def preprocess(self, X, y):
        """
        Preprocess the data by selecting the biggest group of same features for learning.

        Args:
            X: Input features.
            y: Target values.

        Returns:
            Preprocessed X and y.
        """
        Xy = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        features = pd.DataFrame(Xy)
        indices = list(range(len(X[0])))
        indices.remove(self.instance_count_index)
        groups = features.groupby(by=indices)
        max_group = sorted(groups, key=lambda x: len(x[1]))[-1][1]
        X = max_group.iloc[:, 0].to_numpy().reshape((-1, 1))
        y = max_group.iloc[:, -1]
        return X, y

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: Input features.
            y: Target values.
        """
        if X.shape[1] > 1:
            X, y = self.preprocess(X, y)

        self.min, self.max = X.min(), X.max()
        self.regressor.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Args:
            X: Input features.

        Returns:
            Predicted values.
        """
        rt_for_min_scaleout = self.regressor.predict(np.array([[self.min]]))
        rt = self.regressor.predict(X)[:, np.newaxis]
        rt[X.flatten() > self.max] = self.regressor.predict(np.array([[self.max]]))
        return (rt / rt_for_min_scaleout).flatten()

class OptimisticModel(BaseEstimator, RegressorMixin):
    """
    Optimistic modeling combining scale-out speedup and inputs-behavior models.
    """

    def __init__(self, ibm, ssm):
        """
        Initialize the OptimisticModel.

        Args:
            ibm: The inputs-behavior model.
            ssm: The scale-out speedup model.
        """
        self.ssm = SSModel(regressor=ssm)
        self.ibm = ibm

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: Input features.
            y: Target values.
        """
        X, y = np.array(X), np.array(y)
        self.instance_count_index = 0
        self.ssm.fit(X, y)
        scales = self.ssm.predict(X[:, [self.instance_count_index]])
        y_projection = y / scales
        inputs = [i for i in range(X.shape[1]) if i != self.instance_count_index] or [0]
        self.ibm.fit(X[:, inputs], y_projection)

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Args:
            X: Input features.

        Returns:
            Predicted values.
        """
        X = np.array(X)
        instance_count = X[:, [self.instance_count_index]]
        inputs = list([i for i in range(X.shape[1]) if i != self.instance_count_index])
        m1 = self.ssm.predict(instance_count).flatten()
        m2 = self.ibm.predict(X[:, inputs]).flatten()
        y_pred = m1 * m2
        return y_pred

class BasicOptimisticModel(BaseEstimator, RegressorMixin):
    """
    Basic version of the OptimisticModel using Linear Regression and PolynomialFeatures.
    """

    def __init__(self):
        polyreg3 = make_pipeline(PolynomialFeatures(3), LinearRegression())
        self.estimator = OptimisticModel(ibm=LinearRegression(), ssm=polyreg3)
        self.fit = self.estimator.fit
        self.predict = self.estimator.predict

class OptimisticGradientBoosting(BaseEstimator, RegressorMixin):
    """
    Optimistic model using Gradient Boosting for scale-out speedup and inputs-behavior models.
    """

    def __init__(self):
        ssm = GradientBoosting(learning_rate=0.5, n_estimators=50)
        ibm = GradientBoosting(learning_rate=0.05, n_estimators=300)
        self.estimator = OptimisticModel(ibm=ibm, ssm=ssm)
        self.fit = self.estimator.fit
        self.predict = self.estimator.predict

class GradientBoosting(BaseEstimator, RegressorMixin):
    """
    Wrapper class for scikit-learn's GradientBoostingRegressor.
    """

    def __init__(self, learning_rate=0.1, n_estimators=1000):
        """
        Initialize the GradientBoosting model.

        Args:
            learning_rate (float): Learning rate for gradient boosting.
            n_estimators (int): Number of boosting stages.
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        regressor = GradientBoostingRegressor(learning_rate=self.learning_rate,
                                              n_estimators=self.n_estimators)
        estimator = Pipeline(steps=[
            ('ss', StandardScaler()),
            ('gb', regressor)
        ])
        self.fit = estimator.fit
        self.predict = estimator.predict

class ErnestModel(BaseEstimator, RegressorMixin):
    """
    Ernest model using a custom mapping and non-negative least squares optimization.
    """

    def _fmap(self, x):
        """
        Custom feature mapping for the Ernest model.

        Args:
            x: Input features.

        Returns:
            Feature-mapped inputs.
        """
        x = np.array(x)
        scaleout, problem_size = x[:, 0], x[:, 1]
        return np.c_[np.ones_like(scaleout),
                     problem_size / scaleout,
                     np.log(scaleout),
                     scaleout]

    def fit(self, x, y):
        """
        Fit the model to the data.

        Args:
            x: Input features.
            y: Target values.
        """
        X = self._fmap(x)
        y = np.array(y).flatten()
        self.coeff, _ = sp.optimize.nnls(X, y)

    def predict(self, x):
        """
        Make predictions using the fitted model.

        Args:
            x: Input features.

        Returns:
            Predicted values.
        """
        X = self._fmap(x)
        return np.dot(X, self.coeff)
