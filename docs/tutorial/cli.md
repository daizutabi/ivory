# Command Line Interface

If you define data and model, and prepare a YAML parameter file, you don't need to write another Python script code to invoke runs. Ivory's command line interface can do it.

For cross validation:

~~~bash terminal
$ ivory run torch fold=0-4
~~~

For grid search:

~~~bash terminal
$ ivory run torch dropout=0-0.5:5 hidden_sizes.0=10-20-2
~~~

For optimization using a suggest function:

~~~bash terminal
$ ivory optimize torch lr
~~~

For parametric optimization:

~~~bash terminal
$ ivory optimize torch lr=1e-5_1e-3.log
~~~

Right-hand side string for each parameter creates a [Range instance](../task#range) to determine the range of parameters.
