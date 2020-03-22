import os
import urllib.parse
import urllib.request


def to_uri(path: str) -> str:
    """
    Examples:
        >>> path = "abc\\def"
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
