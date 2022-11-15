from pathlib import Path

from yieldprediction.experiment import ExperimentConfig


config = ExperimentConfig(n_resamples=1, 
                          model_configs=[
                                         #Path('configs/model_config-molecular_descriptor_random_forest.py'),
                                         #Path('configs/model_config-molecular_descriptor_linear_regression.py'),
                                         Path('configs/model_config-id_based_catboost.py'),
                                         ], 
                          datasets=[Path('datasets/suzuki/processed_nodupes_dataset_conf.py')], 
                          #datasets=[Path('datasets/buchwaldhartwig/processed_nodupes_dataset_conf.py')], 
                          #datasets=[Path('datasets/uspto/processed_nodupes_dataset_conf.py')], 
                          test_ratio=0.1, 
                          random_seed=1729)