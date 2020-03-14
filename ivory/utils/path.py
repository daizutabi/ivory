import os


def to_uri(path: str) -> str:
    if ":" in path:
        raise NotImplementedError
    if "~" in path:
        path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if "\\" in path:
        path = "/" + path.replace("\\", "/")
    return "file://" + path
