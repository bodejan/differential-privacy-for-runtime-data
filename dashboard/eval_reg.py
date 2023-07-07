


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
    def eval_input_data(input_data):
        def get_training_data(input_data):
            def get_features_filters(input_data):
                if 'sort' in input_data:
                    return ['data_size_MB'], [('machine_type', '==', 'c4.2xlarge'),('line_length', '==', 100)]
                elif 'grep' in input_data:
                    return ['data_size_MB', 'p_occurrence'], [('machine_type', '==', 'm4.2xlarge')]
                elif 'sgd' in input_data:
                    return ['observations', 'features', 'iterations'],[('machine_type', '==', 'r4.2xlarge'),('instance_count', '>', 2)]
                elif 'kmeans' in input_data:
                    return ['observations', 'features', 'k'],[('machine_type', '==', 'r4.2xlarge'),('instance_count', '>', 2)]
                elif 'page' or 'rank' in input_data:
                    return ['links', 'pages', 'convergence_criterion'], [('machine_type', '==', 'r4.2xlarge')]
                else:
                    return None

            features, filters = get_features_filters(input_data)
            input_df = pd.read_csv(input_data)
            g = input_df.groupby(by=['instance_count','machine_type']+features)
            input_df = pd.DataFrame(g.median().to_records())
            # Apply filters
            # e.g. only for one machine type each, the full c3o-experiments were conducted
            # No full cartesian product!
            for k, s, v in filters:
                if s == '==': input_df = input_df[input_df[k] == v]
                if s == '>' : input_df = input_df[input_df[k] >  v]
            X = input_df[['instance_count'] + features]
            y = (input_df[['gross_runtime']]).squeeze()
            return X, y
        
        def init_models():
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

        def pred(model, X, y, test_size: float=0.1):
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size)
            model.fit(X_tr, y_tr)
            y_hat = model.predict(X_te)
            errors = (y_hat - y_te).to_numpy()
            mse, std = errors.mean()**2, errors.std()
            mape = np.mean(np.abs(errors / y_te)) * 100
            return mse, std, mape
        

        X,y = get_training_data(input_data)
        results = []
        model_names = ['GradientBoosting', 'ErnestModel', 'OptimisticGradientBoosting','BasicOptimisticModel']
        models = init_models()
        for model_name, model in zip(model_names, models):
            mse, std, mape = pred(model, X, y)
            result = {'name': model_name, 'mse': mse, 'std': std, 'mape': mape}
            results.append(result)
            #print(f'Predicting {model_name}: {result}')
        return results
    

class SSModel(BaseEstimator, RegressorMixin):  # Scaleout-Speedup model

    def __init__(self, instance_count_index=0, regressor=None):
        self.regressor = regressor
        self.scales = []
        self.instance_count_index = instance_count_index  # index within features

    def preprocess(self, X, y):
        # Find biggest group of same features besides instance_count to learn from
        Xy = np.concatenate((X, y.reshape(-1,1)), axis=1)
        features = pd.DataFrame(Xy)
        indices = list(range(len(X[0])))
        indices.remove(self.instance_count_index)
        groups = features.groupby(by=indices)
        max_group = sorted(groups, key=lambda x:len(x[1]))[-1][1]
        X = max_group.iloc[:, 0].to_numpy().reshape((-1,1))
        y = max_group.iloc[:, -1]
        return X, y

    def fit(self, X, y):
        if X.shape[1] > 1:
            X, y = self.preprocess(X, y)

        self.min, self.max = X.min(), X.max()
        self.regressor.fit(X,y)

    def predict(self, X):
        rt_for_min_scaleout = self.regressor.predict(np.array([[self.min]]))
        # Make it a 2-dim array, as it is usually supposed to be
        rt = self.regressor.predict(X)[:, np.newaxis]
        # Replace scale-outs of more than self.max with pred for self.max
        # (poly3 curve does not continue as desired)
        rt[X.flatten() > self.max] = self.regressor.predict(np.array([[self.max]]))
        return (rt/rt_for_min_scaleout).flatten()


class OptimisticModel(BaseEstimator, RegressorMixin):

    def __init__(self, ibm, ssm):
        self.ssm= SSModel(regressor=ssm)
        self.ibm= ibm

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.instance_count_index = 0
        # Train scale-out speed-up model
        self.ssm.fit(X, y)
        scales = self.ssm.predict(X[:,[self.instance_count_index]])
        #print('scales', scales.shape)
        # Project all runtimes to expected runtimes at scaleout = min_scaleout
        y_projection = y/scales
        #print('yproj', y_projection.shape)
        # Train the inputs-behavior model on all inputs (not the instance_count)
        inputs = [i for i in range(X.shape[1]) if i != self.instance_count_index] or [0]
        self.ibm.fit(X[:,inputs], y_projection)

    def predict(self, X):
        X = np.array(X)
        instance_count = X[:, [self.instance_count_index]]
        inputs = list([i for i in range(X.shape[1]) if i != self.instance_count_index])
        m1 = self.ssm.predict(instance_count).flatten()
        m2 = self.ibm.predict(X[:,inputs]).flatten()
        y_pred = m1 * m2
        return y_pred


class BasicOptimisticModel(BaseEstimator, RegressorMixin):

    def __init__(self):
        polyreg3 = make_pipeline(PolynomialFeatures(3), LinearRegression())
        self.estimator = OptimisticModel(ibm=LinearRegression(), ssm=polyreg3)

        self.fit = self.estimator.fit
        self.predict = self.estimator.predict


class OptimisticGradientBoosting(BaseEstimator, RegressorMixin):

    def __init__(self):
        ssm = GradientBoosting(learning_rate=0.5, n_estimators=50)
        ibm = GradientBoosting(learning_rate=0.05, n_estimators=300)
        self.estimator = OptimisticModel(ibm=ibm, ssm=ssm)

        self.fit = self.estimator.fit
        self.predict = self.estimator.predict


class GradientBoosting(BaseEstimator, RegressorMixin):

    def __init__(self, learning_rate=0.1, n_estimators=1000):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        regressor = GradientBoostingRegressor(learning_rate=self.learning_rate,
                                              n_estimators=self.n_estimators)
        estimator = Pipeline(steps=[
                ('ss', StandardScaler()),
                ('gb', regressor) ])

        self.fit = estimator.fit
        self.predict = estimator.predict


class ErnestModel(BaseEstimator, RegressorMixin):

    def _fmap(self, x):
        x = np.array(x)
        scaleout, problem_size = x[:,0], x[:,1]
        return np.c_[np.ones_like(scaleout),
                     problem_size/scaleout,
                     np.log(scaleout),
                     scaleout]

    def fit(self, x, y):
        X = self._fmap(x)
        y = np.array(y).flatten()
        self.coeff, _ = sp.optimize.nnls(X, y)

    def predict(self, x):
        X = self._fmap(x)
        return np.dot(X, self.coeff)


if __name__ == '__main__':
    results_original = Regression.eval_input_data('datasets/sort.csv')
    print(results_original, '\n', '\n')
    results_synthetic = Regression.eval_input_data('dashboard/temp/sort_synthetic_123.csv')
    print(results_synthetic)