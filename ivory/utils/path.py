import contextlib
import os
import urllib.parse
import urllib.request


def to_uri(path: str) -> str:
    """
    Examples:
        >>> path = r"abc\\def"
        >>> to_uri(path)
        'file:///abc/def'
    """
    if urllib.parse.urlparse(path).scheme:
        return path
    if "~" in path:
        path = os.path.expanduser(path)
    url = os.path.abspath(path)
    if "\\" in url:
        url = urllib.request.pathname2url(path)
    return urllib.parse.urlunparse(("file", "", url, "", "", ""))


@contextlib.contextmanager
def chdir(source_name):
    curdir = os.getcwd()
    if source_name:
        basedir = os.path.dirname(source_name)
        os.chdir(basedir)
    try:
        yield
    finally:
        os.chdir(curdir)
