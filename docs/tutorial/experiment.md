# Experiment

The first start point in Ivory is `ivory.Experiment`. This master object controls a machine learning experiment under a certain condition. Let's create your first `Experiment` instance.

In your favorite directory, write a YAML file like below named `params_0.yaml`:

#File params_0.yaml {%=params_0.yaml%}

The `class` key notices that the `experiment` is an instance of `ivory.Experiment`.

Then, in a jupyte notebook or a Python script under the same directory:

```python
import ivory

experiment = ivory.create_experiment('params_0.yaml')
experiment
```

You created an `Experiment` instance. It says that its name is "ready" and its `run_class` is `ivory.core.Run`. Next, you can start the `Experiment`.

```python
experiment.start()
experiment
```

The name was changed to the current time.

An `Experiment` instance is like an environment. Actual interesting processes such as machine learning are done by `Run`s. So, create a `Run`.

```python
run = experiment.create_run()
run
```

This `Run` instance was named '#1' because this is the first one for the `Experiment` instance.

Do you want to start this run? Let's try it.

```python
run.start()
```

Oops! The `Run` instance has the method indeed, but it says there is no `trainer`. Of course, we need some data, a model, metrics, *etc*. for a particular machine learning problem. Also, we need to decide how to train the model. Following sections will explain this process step by step.
