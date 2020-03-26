from typing import Dict

from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.utils.mlflow_tags import (MLFLOW_GIT_COMMIT, MLFLOW_RUN_NAME,
                                      MLFLOW_SOURCE_NAME)

cache: Dict[str, str] = {}


def get_tags(name: str, source_name: str = ""):
    tags = {MLFLOW_RUN_NAME: name}
    if source_name:
        tags[MLFLOW_SOURCE_NAME] = source_name
        if source_name not in cache:
            cache[source_name] = _get_git_commit(source_name)
        if cache[source_name]:
            tags[MLFLOW_GIT_COMMIT] = cache[source_name]
    tags = context_registry.resolve_tags(tags)
    return tags


def get_source_name(run):
    return run.data.tags[MLFLOW_SOURCE_NAME]
