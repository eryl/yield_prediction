Experiments on yield prediction
===============================
This repository contains the code to carry out experiments on yield prediction.


Dependencies
============
All dependencies are handled through Anaconda and were developed with Anaconda 4.10.1. After installing anaconda, initialize the environment for this repository using:

```
$> conda env create -f environment.yml
```

After this is done, initialize the environment by running:

```
$> conda activate yield-prediction
```

Code organization
=================
The experiments are organized around python scripts. Experiments are configured using python script files (see the directory `configs/`). The script to run experiments is `scripts/run_experiment.py`. Run it from a command line like so:

```
$> python scripts/run_experiment.py configs/yield_prediction_experiment.py
```

All settings for experiments are configured in this file, which in turn will use other configuration files for the different models to run.

Explaining datasets
===================

The code uses a modular approach to datasets. Datasets are described using a special python file containing an instance of the `DatasetSpec` class which describes to the experiment code how the dataset should be loaded. The experiment configuration file refers to this python file. To add a new dataset, you can copy one of the existing files (see `datasets/uspto/raw_dataset_conf.py` for an example).