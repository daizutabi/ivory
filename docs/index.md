# Ivory

## Overview

Ivory is a lightweight framework for machine learning. It integrates model design, hyperparmeter tuning, and tracking. Ivory uses [Optuna](https://preferred.jp/en/projects/optuna/) for hyperparmeter tuning and [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) for tracking.

The relationship of these libraries is like below:

* Ivory's `Experiment` = Optuna's `Study` = MLflow's `Experiment`
* Ivory's `Run` = Optuna's `Trial` = MLflow's `Run`

Using Ivory, you can obtain the both tuning and tracking workflow at one place.

Another key feature of Ivory is its model design. You can write down all of your model structure and tuning/tracking process in one YAML file. Its allows us to understand the whole process at a glance.


## Installation

You can install Ivory from PyPI.

~~~bash terminal
$ pip install ivory
~~~
