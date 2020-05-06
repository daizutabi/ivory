import sklearn.ensemble
import sklearn.linear_model

import ivory.core.estimator


class Estimator(ivory.core.estimator.Estimator):
    def __init__(self, estimator_factory, return_probability=True, **kwargs):
        super().__init__(estimator_factory, **kwargs)
        self.return_probability = return_probability
        if self.params:
            raise ValueError(f"Unknown parameters: {list(self.params.keys())}")
        self.estimator = estimator_factory(**self.kwargs)

    def predict(self, input):
        if self.return_probability:
            return self.estimator.predict_proba(input)
        else:
            return self.estimator.predict(input)


class Ridge(Estimator):
    def __init__(self, **kwargs):
        super().__init__(sklearn.linear_model.Ridge, return_probability=False, **kwargs)


class RandomForestClassifier(Estimator):
    def __init__(self, **kwargs):
        super().__init__(sklearn.ensemble.RandomForestClassifier, **kwargs)


class RandomForestRegressor(Estimator):
    def __init__(self, **kwargs):
        super().__init__(
            sklearn.ensemble.RandomForestRegressor, return_probability=False, **kwargs
        )
