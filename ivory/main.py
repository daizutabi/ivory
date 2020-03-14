import argparse

import ivory.core


class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("params", nargs="*")
        parser.add_argument("-p", "--params-path", default="params.yaml")
        args = parser.parse_args()
        experiment = ivory.core.experiment.create_experiment(args.params_path)
        print(type(experiment.module))
        print(experiment)
        experiment.start()
        run = experiment.create_run()
        print(run.input)


def main():
    Parser()
