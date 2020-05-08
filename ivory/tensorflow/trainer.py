import ivory.core.estimator


class Trainer(ivory.core.estimator.Trainable):
    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(self.model.fit, **kwargs)
        if self.params:
            raise ValueError(f"Unknown parameters: {list(self.params.keys())}")

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}()"

    def fit(self, input, target, val=None):
        if val is not None:
            val = tuple(val)
        self.model.fit(input, target, validation_data=val, **self.kwargs)

    def predict(self, input):
        return self.model.predict(input)
