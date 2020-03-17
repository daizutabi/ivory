__version__ = "0.1.2"

from ivory.core import instance

create_instance = instance.create_instance
create_environment = instance.create_instance_factory("environment")
create_experiment = instance.create_instance_factory("experiment")
create_run = instance.create_instance_factory("run")
