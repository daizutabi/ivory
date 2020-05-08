import ivory.core.run


class Run(ivory.core.run.Run):
    def save_instance(self, state_dict, path):
        pass
        # torch.save(state_dict, path)

    def load_instance(self, path):
        pass
        # return torch.load(path)
