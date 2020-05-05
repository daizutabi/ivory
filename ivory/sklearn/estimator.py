import sklearn.ensemble
import sklearn.linear_model

import ivory.core.estimator


class Estimator(ivory.core.estimator.Estimator):
    __estimator__ = None

    def __init__(self, return_probability=True, **kwargs):
        super().__init__(**kwargs)
        self.return_probability = return_probability
        if self.params:
            raise ValueError(f"Unknown parameters: {list(self.params.keys())}")
        self.estimator = self.__estimator__(**self.kwargs)

    def predict(self, input):
        if self.return_probability:
            return self.estimator.predict_proba(input)
        else:
            return self.estimator.predict(input)


class Ridge(Estimator):
    __estimator__ = sklearn.linear_model.Ridge

    def __init__(self, **kwargs):
        super().__init__(return_probability=False, **kwargs)


class RandomForestClassifier(Estimator):
    __estimator__ = sklearn.ensemble.RandomForestClassifier
