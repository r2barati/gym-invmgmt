"""Shim for editable installs with older pip versions."""
from setuptools import setup, find_packages

setup(
    packages=find_packages(include=["gym_invmgmt*"]),
)
