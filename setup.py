import os
from setuptools import setup, find_packages

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(PACKAGE_ROOT, 'requirements.txt'), 'r') as read_file:
    REQUIREMENTS = read_file.read().splitlines()

setup(
    name='ml-models',
    version='0.2.1',
    description='ML Models',
    author="Daniel Poon",
    url="http://github.com/danielpoonwj/ml-models",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    zip_safe=False
)
