# File: setup.py

from setuptools import find_packages, setup

setup(
    name="cool_science",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)