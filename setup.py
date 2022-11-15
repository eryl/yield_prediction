# coding: utf-8

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='yieldprediction',
      version='0.1',
      description='Experiments on reaction yield prediction',
      url='https://github.com/eryl/yieldprediction',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['yieldprediction'],
      install_requires=[],
      dependency_links=[],
      zip_safe=False)