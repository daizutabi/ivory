import sklearn.ensemble
import sklearn.linear_model

import ivory.core.estimator
from ivory.core import instance


class Estimator(ivory.core.estimator.Estimator):
    def __init__(self, model, return_probability=True, **kwargs):
        if isinstance(model, str):
            model = instance.get_attr(model)
        super().__init__(model, **kwargs)
        if self.params:
            raise ValueError(f"Unknown parameters: {list(self.params.keys())}")
        self.estimator = model(**self.kwargs)
        if not hasattr(self.estimator, "predict_proba"):
            return_probability = False
        self.return_probability = return_probability

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
