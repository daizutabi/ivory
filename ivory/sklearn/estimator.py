import ivory.core.estimator
from ivory.core import instance


class Estimator(ivory.core.estimator.Estimator):
    def __init__(self, model, return_probability=True, **kwargs):
        model = instance.get_attr(model)
        params, kwargs = instance.filter_params(model, **kwargs)
        if params:
            raise ValueError(f"Unknown parameters: {list(params.keys())}")
        self.estimator = model(**kwargs)
        if not hasattr(self.estimator, "predict_proba"):
            return_probability = False
        self.return_probability = return_probability

    def predict(self, input):
        if self.return_probability:
            return self.estimator.predict_proba(input)
        else:
            return self.estimator.predict(input)
