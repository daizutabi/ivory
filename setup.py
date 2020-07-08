import os
import re
import subprocess
import sys

from setuptools import setup


def get_version(package: str) -> str:
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), package, "__init__.py"
    )
    with open(path, "r") as file:
        source = file.read()
    m = re.search(r'__version__ = ["\'](.+)["\']', source)
    if m:
        return m.group(1)
    else:
        return "0.0.0"


def get_packages(package):
    """Return root package and all sub-packages."""
    return [
        dirpath
        for dirpath, dirnames, filenames in os.walk(package)
        if os.path.exists(os.path.join(dirpath, "__init__.py"))
    ]


def check():
    def run(command):
        assert subprocess.run(command.split()).returncode == 0
        print(f"'{command}' --- OK")

    run("pycodestyle ivory")
    run("pyflakes ivory")
    run("mypy ivory")
    run("pycodestyle tests")
    run("pyflakes tests")
    run("mypy tests")


def publish():
    check()
    subprocess.run("python setup.py sdist bdist_wheel".split())
    subprocess.run("twine upload dist/*".split())
    version = get_version("ivory")
    subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"'Version {version}'"])
    subprocess.run(["git", "push", "origin", "--tags"])
    sys.exit(0)


if sys.argv[-1] == "publish":
    publish()

if sys.argv[-1] == "check":
    check()


long_description = "ML Tool."

setup(
    name="ivory",
    version=get_version("ivory"),
    description="ML Tool",
    long_description=long_description,
    url="https://ivory.daizutabi.net",
    author="daizutabi",
    author_email="daizutabi@gmail.com",
    license="MIT",
    packages=get_packages("ivory"),
    include_package_data=True,
    entry_points={"console_scripts": ["ivory = ivory.main:main"]},
    install_requires=[
        "mlflow",
        "optuna",
        "numpy",
        "pandas",
        "scikit-learn",
        "termcolor",
        "click",
        "logzero",
        "tqdm",
    ],
    classifiers=[
        "Environment :: Console",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
