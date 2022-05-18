"""
Basic setup file for connections library
"""

from setuptools import setup, find_packages

setup(
    name='tfs-attribution',
    # Hard code for now
    version='0.0.1',
    description='Marketing attribution tools',
    author='LSLPG Data Science',
    author_email='chris.bishop@thermofisher.com',
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='marketing Shapley attribution',
    url='https://github.com/chris-bishop-tfs/data_io',
    packages=['attribution'],
    install_requires=[
      'pyspark>=3.2.1',
      'attrs'
    ]
)
