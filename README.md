[![PyPI version][pypi-image]][pypi-link]
[![Python versions][pyversions-image]][pyversions-link]
[![Travis][travis-image]][travis-link]
[![Coverage Status][coveralls-image]][coveralls-link]
[![Code style: black][black-image]][black-link]

# Ivory

Ivory is a lightweight framework for machine learning. It integrates model design, hyperparmeter tuning, and tracking. Ivory uses [Optuna](https://preferred.jp/en/projects/optuna/) for hyperparmeter tuning and [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) for tracking.

The relationship of these libraries is like below:

* Ivory's `Experiment` = Optuna's `Study` = MLflow's `Experiment`
* Ivory's `Run` = Optuna's `Trial` = MLflow's `Run`

Using Ivory, you can obtain the both tuning and tracking workflow at one place.

Another key feature of Ivory is its model design. You can write down all of your model structure and tuning/tracking process in one YAML file. It allows us to understand the whole process at a glance.


## Setup

Install Ivory using pip:

```bash
pip install ivory
```

## Documentation

See [Ivory documentation](https://ivory.daizutabi.net).

## Overview

![overview](overview.png "Overview of Ivory components")


[pypi-image]: https://badge.fury.io/py/ivory.svg
[pypi-link]: https://pypi.org/project/ivory
[travis-image]: https://travis-ci.org/daizutabi/ivory.svg?branch=master
[travis-link]: https://travis-ci.org/daizutabi/ivory
[coveralls-image]: https://coveralls.io/repos/github/daizutabi/ivory/badge.svg?branch=master
[coveralls-link]: https://coveralls.io/github/daizutabi/ivory?branch=master
[black-image]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/ambv/black
[pyversions-image]: https://img.shields.io/pypi/pyversions/ivory.svg
[pyversions-link]: https://pypi.org/project/ivory
