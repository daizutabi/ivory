import nnabla as nn

import ivory.core.collections


class Model:
    NUM_MODELS = 0

    def __init__(self):
        self.training = True
        self.scope = f"model{self.__class__.NUM_MODELS}"
        self.__class__.NUM_MODELS += 1

    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.train(False)

    def build(self, loss, dataset, batch_size):
        index, input, target = dataset[0]
        input_shape = [batch_size] + list(input.shape)
        target_shape = [batch_size] + list(target.shape)
        self.input = ivory.core.collections.Dict()
        self.output = ivory.core.collections.Dict()
        self.target = ivory.core.collections.Dict()
        self.loss = ivory.core.collections.Dict()
        for mode in ["train", "test"]:
            self.input[mode] = nn.Variable(input_shape)
            self.target[mode] = nn.Variable(target_shape)
            with nn.parameter_scope(self.scope):
                self.training = mode == "train"
                self.output[mode] = self.forward(self.input[mode])
            if mode == "train":
                self.output[mode].persistent = True
            self.loss[mode] = loss(self.output[mode], self.target[mode])
        self.training = True

    def parameters(self):
        with nn.parameter_scope(self.scope):
            return nn.get_parameters()

    def forward(self, input):
        raise NotImplementedError

    def __call__(self, input, target=None):
        if self.training:
            mode = "train"
        else:
            mode = "test"
        self.input[mode].data.data = input
        if target is not None:
            self.target[mode].data.data = target
            node = self.loss[mode]
        else:
            node = self.output[mode]
        if mode == "train":
            node.forward()  # clear_no_need_grad=True)
        else:
            node.forward()  # clear_buffer=True)
        output = self.output[mode].data.data.copy()
        if target is None:
            return output
        else:
            return output, self.loss[mode].data.data.copy()

    def backward(self):
        self.loss["train"].backward()  # clear_buffer=True)
