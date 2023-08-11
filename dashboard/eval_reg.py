from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import RegressorMixin, BaseEstimator


class Regression():
    """
    A class for performing regression analysis using various models and evaluating their performance.

    Methods:
        eval_input_data(input_file, dataset_name, split, original_file=None):
            Evaluate input data using different regression models.
    """
    @staticmethod
    def eval_input_data(input_file: str, dataset_name: str, split, original_file: str = None):
        """
        Evaluate input data using different regression models.

        Args:
            input_file (str): Path to the input CSV file containing data.
            dataset_name (str): Name of the dataset being used.
            split: Proportion of data to use for testing (between 0 and 1).
            original_file (str, optional): Path to the original input file for synthetic data (default: None).

        Returns:
            list: List of dictionaries containing model names and their evaluation results.
        """
        def get_training_data(file):
            """
            Load and preprocess training data based on the dataset name.

            Args:
                file (str): Path to the input CSV file.

            Returns:
                pandas.DataFrame: Feature matrix (X) and target vector (y).
            """
            # original_file when training on synthetic data
            def get_features_filters():
                if dataset_name == 'sort':
                    return ['data_size_MB'], [('machine_type', '==', 'c4.2xlarge'), ('line_length', '==', 100)]
                elif dataset_name == 'grep':
                    return ['data_size_MB', 'p_occurrence'], [('machine_type', '==', 'm4.2xlarge')]
                elif dataset_name == 'sgd':
                    return ['observations', 'features', 'iterations'], [('machine_type', '==', 'r4.2xlarge'),
                                                                        ('instance_count', '>', 2)]
                elif dataset_name == 'kmeans':
                    return ['observations', 'features', 'k'], [('machine_type', '==', 'r4.2xlarge'),
                                                               ('instance_count', '>', 2)]
                elif dataset_name == 'page' or dataset_name == 'rank':
                    return ['links', 'pages', 'convergence_criterion'], [('machine_type', '==', 'r4.2xlarge')]
                else:
                    return None

            features, filters = get_features_filters()
            input_df = pd.read_csv(file)
            g = input_df.groupby(by=['instance_count', 'machine_type'] + features)
            input_df = pd.DataFrame(g.median().to_records())
            # Apply filters
            # e.g. only for one machine type each, the full c3o-experiments were conducted
            # No full cartesian product!
            for k, s, v in filters:
                if s == '==': input_df = input_df[input_df[k] == v]
                if s == '>': input_df = input_df[input_df[k] > v]
            X = input_df[['instance_count'] + features]
            y = (input_df[['gross_runtime']]).squeeze()
            return X, y

        def init_models():
            """
            Initialize different regression models.

            Returns:
                tuple: Instances of different regression models.
            """
            # default models
            # GradientBoostingRegressor 
            gb = GradientBoosting()
            # Model for performance predictions using 'nnls'
            em = ErnestModel()
            # custom models
            # ibm & ssm: GradientBoosting
            ogb = OptimisticGradientBoosting()
            # ibm: LinearRegression, ssm: polyreg3
            bom = BasicOptimisticModel()
            return gb, em, ogb, bom

        def pred(model, X, y, test_size: float = 0.1, X_original = None, y_original = None):
            """
            Make predictions using a given model and evaluate its performance.

            Args:
                model: A regression model instance.
                X: Feature matrix.
                y: Target vector.
                test_size (float, optional): Proportion of data to use for testing (default: 0.1).
                X_original: Feature matrix of original data (default: None).
                y_original: Target vector of original data (default: None).

            Returns:
                tuple: Mean squared error (mse), standard deviation (std), mean absolute percentage error (mape).
            """
            # evaluate using the original data, if provided
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42, test_size=test_size)
            if X_original:
                X_tr_original, X_te_original, y_tr_original, y_te_original = train_test_split(X_original, y_original, random_state=42, test_size=test_size)
                model.fit(X_tr, y_tr)
                y_hat = model.predict(X_te_original)
                errors = (y_hat - y_te_original).to_numpy()
            else:
                model.fit(X_tr, y_tr)
                y_hat = model.predict(X_te)
                errors = (y_hat - y_te).to_numpy()
            mse, std = errors.mean() ** 2, errors.std()
            mape = np.mean(np.abs(errors / y_te)) * 100
            return mse, std, mape

        X, y = get_training_data(input_file)
        if original_file:
            X_orginal, y_original = get_training_data(original_file)
        results = []
        model_names = ['GradientBoosting', 'ErnestModel', 'OptimisticGradientBoosting', 'BasicOptimisticModel']
        models = init_models()
        for model_name, model in zip(model_names, models):
            mse, std, mape = pred(model, X, y, 1 - split)
            result = {'name': model_name, 'mse': mse, 'std': std, 'mape': mape}
            results.append(result)
            # print(f'Predicting {model_name}: {result}')
        return results


class SSModel(BaseEstimator, RegressorMixin):  # Scaleout-Speedup model
    """
    Scaleout-Speedup Model (SSModel) for predicting runtime based on instance count and other features.
    Inherits from BaseEstimator and RegressorMixin.

    Args:
        instance_count_index (int, optional): Index of the instance count feature (default: 0).
        regressor: Regression model to use for scale-out predictions.

    Methods:
        preprocess(X, y):
            Preprocesses the input data to extract the largest group of same features besides instance_count.
        fit(X, y):
            Fits the model to the training data.
        predict(X):
            Predicts runtimes based on instance count and other features.
    """

    def __init__(self, instance_count_index=0, regressor=None):
        self.regressor = regressor
        self.scales = []
        self.instance_count_index = instance_count_index  # index within features

    def preprocess(self, X, y):
        """
        Preprocesses the input data to extract the largest group of same features besides instance_count.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Preprocessed feature matrix (X) and target vector (y).
        """
        # Find biggest group of same features besides instance_count to learn from
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
        Fits the model to the training data.

        Args:
            X: Feature matrix.
            y: Target vector.
        """
        if X.shape[1] > 1:
            X, y = self.preprocess(X, y)

        self.min, self.max = X.min(), X.max()
        self.regressor.fit(X, y)

    def predict(self, X):
        """
        Predicts runtimes based on instance count and other features.

        Args:
            X: Feature matrix.

        Returns:
            numpy.ndarray: Predicted runtimes.
        """
        rt_for_min_scaleout = self.regressor.predict(np.array([[self.min]]))
        # Make it a 2-dim array, as it is usually supposed to be
        rt = self.regressor.predict(X)[:, np.newaxis]
        # Replace scale-outs of more than self.max with pred for self.max
        # (poly3 curve does not continue as desired)
        rt[X.flatten() > self.max] = self.regressor.predict(np.array([[self.max]]))
        return (rt / rt_for_min_scaleout).flatten()


class OptimisticModel(BaseEstimator, RegressorMixin):
    """
    Optimistic Model for predicting runtime based on Scaleout-Speedup Model and behavior model.
    Inherits from BaseEstimator and RegressorMixin.

    Args:
        ibm: Behavior model for input features.
        ssm: Scaleout-Speedup Model for predicting scale-out effects.

    Methods:
        fit(X, y):
            Fits the model to the training data.
        predict(X):
            Predicts runtimes based on input features.
    """

    def __init__(self, ibm, ssm):
        self.ssm = SSModel(regressor=ssm)
        self.ibm = ibm

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X: Feature matrix.
            y: Target vector.
        """
        X, y = np.array(X), np.array(y)
        self.instance_count_index = 0
        # Train scale-out speed-up model
        self.ssm.fit(X, y)
        scales = self.ssm.predict(X[:, [self.instance_count_index]])
        # print('scales', scales.shape)
        # Project all runtimes to expected runtimes at scaleout = min_scaleout
        y_projection = y / scales
        # print('yproj', y_projection.shape)
        # Train the inputs-behavior model on all inputs (not the instance_count)
        inputs = [i for i in range(X.shape[1]) if i != self.instance_count_index] or [0]
        self.ibm.fit(X[:, inputs], y_projection)

    def predict(self, X):
        """
        Predicts runtimes based on input features.

        Args:
            X: Feature matrix.

        Returns:
            numpy.ndarray: Predicted runtimes.
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
    Basic Optimistic Model that combines Linear Regression and Polynomial Regression.

    This model uses the OptimisticModel to make predictions by combining Linear Regression for input behavior modeling
    and Polynomial Regression for scale-out speed-up modeling.

    Attributes:
        estimator: An instance of the OptimisticModel using Linear Regression and Polynomial Regression.

    Methods:
        fit(X, y):
            Fits the model to the training data.
        predict(X):
            Predicts runtimes based on input features.
    """

    def __init__(self):
        polyreg3 = make_pipeline(PolynomialFeatures(3), LinearRegression())
        self.estimator = OptimisticModel(ibm=LinearRegression(), ssm=polyreg3)

        self.fit = self.estimator.fit
        self.predict = self.estimator.predict


class OptimisticGradientBoosting(BaseEstimator, RegressorMixin):
    """
    Optimistic Gradient Boosting Model for predicting runtime based on OptimisticModel with Gradient Boosting.

    This model uses the OptimisticModel with Gradient Boosting regressors for input behavior modeling
    and scale-out speed-up modeling.

    Attributes:
        estimator: An instance of the OptimisticModel using Gradient Boosting regressors.

    Methods:
        fit(X, y):
            Fits the model to the training data.
        predict(X):
            Predicts runtimes based on input features.
    """

    def __init__(self):
        ssm = GradientBoosting(learning_rate=0.5, n_estimators=50)
        ibm = GradientBoosting(learning_rate=0.05, n_estimators=300)
        self.estimator = OptimisticModel(ibm=ibm, ssm=ssm)

        self.fit = self.estimator.fit
        self.predict = self.estimator.predict


class GradientBoosting(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting Model for predicting runtime.

    This model uses the GradientBoostingRegressor with feature scaling.

    Args:
        learning_rate (float, optional): Learning rate for gradient boosting (default: 0.1).
        n_estimators (int, optional): Number of boosting stages (default: 1000).

    Attributes:
        learning_rate (float): Learning rate for gradient boosting.
        n_estimators (int): Number of boosting stages.

    Methods:
        fit(X, y):
            Fits the model to the training data.
        predict(X):
            Predicts runtimes based on input features.
    """

    def __init__(self, learning_rate=0.1, n_estimators=1000):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        regressor = GradientBoostingRegressor(learning_rate=self.learning_rate,
                                              random_state=42,
                                              n_estimators=self.n_estimators)
        estimator = Pipeline(steps=[
            ('ss', StandardScaler()),
            ('gb', regressor)])

        self.fit = estimator.fit
        self.predict = estimator.predict


class ErnestModel(BaseEstimator, RegressorMixin):
    """
    Ernest Model for predicting runtime based on feature mapping and non-negative least squares optimization.

    This model applies feature mapping and non-negative least squares optimization to predict runtime.

    Methods:
        fit(x, y):
            Fits the model to the training data.
        predict(x):
            Predicts runtimes based on input features.
    """

    def _fmap(self, x):
        x = np.array(x)
        scaleout, problem_size = x[:, 0], x[:, 1]
        return np.c_[np.ones_like(scaleout),
        problem_size / scaleout,
        np.log(scaleout),
        scaleout]

    def fit(self, x, y):
        """
        Fits the model to the training data.

        Args:
            x: Input features.
            y: Target vector.
        """
        X = self._fmap(x)
        y = np.array(y).flatten()
        self.coeff, _ = sp.optimize.nnls(X, y)

    def predict(self, x):
        """
        Predicts runtimes based on input features.

        Args:
            x: Input features.

        Returns:
            numpy.ndarray: Predicted runtimes.
        """
        X = self._fmap(x)
        return np.dot(X, self.coeff)


if __name__ == '__main__':
    results_original = Regression.eval_input_data('datasets/sort.csv')
    print(results_original, '\n', '\n')
    results_synthetic = Regression.eval_input_data('dashboard/temp/sort_synthetic_123.csv')
    print(results_synthetic)
