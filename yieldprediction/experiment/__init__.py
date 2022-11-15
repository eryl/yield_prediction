from dataclasses import dataclass
from typing import List, Union, Optional
from pathlib import Path

import numpy as np

from yieldprediction.utils import get_module_obj
from yieldprediction.dataset import Dataset
from yieldprediction.models.model import Model
from yieldprediction.experiment.experiment_recorder import ExperimentRecorder


@dataclass
class ExperimentConfig:
    n_resamples: int
    model_configs: List[Path]
    datasets: List[Path]
    test_ratio: float = 0.1
    random_seed: Optional[int] = None

class Experiment:
    def __init__(self, experiment_root: Path, config_path: Path):
        self.experiment_root = experiment_root
        self.config = get_module_obj(config_path, ExperimentConfig)
        self.rng = np.random.default_rng(self.config.random_seed)

    def run_experiments(self):
        for dataset_conf in self.config.datasets:
            dataset = Dataset.from_config(dataset_conf)
            experiment_recorder = ExperimentRecorder(self.experiment_root / dataset.name)
            for i in range(self.config.n_resamples):
                # Standard samplings below
                data_rng = np.random.default_rng(i)
                model_rng = np.random.default_rng(i)
                bootstrap_recorder = experiment_recorder.make_child(f'resample_{i}')
                
                training_data, test_data = dataset.standard_sampling(test_ratio=self.config.test_ratio, rng=data_rng)
                self.fit_and_evaluate_models(bootstrap_recorder.make_child('standard_sampling'), training_data, test_data, data_rng=data_rng, model_rng=model_rng)
                
                training_data, test_data = dataset.balanced_sampling(test_ratio=self.config.test_ratio, rng=data_rng)
                self.fit_and_evaluate_models(bootstrap_recorder.make_child('balanced_sampling'), training_data, test_data, data_rng=data_rng, model_rng=model_rng)
                
                training_data, test_data = dataset.uniques_sampling(test_ratio=self.config.test_ratio, rng=data_rng)
                uniques_train_n = len(training_data)
                uniques_test_n = len(test_data)
                self.fit_and_evaluate_models(bootstrap_recorder.make_child('uniques_sampling'), training_data, test_data, data_rng=data_rng, model_rng=model_rng)

                training_data, test_data = dataset.standard_sampling(test_n=uniques_test_n, train_n=uniques_train_n, rng=data_rng)
                self.fit_and_evaluate_models(bootstrap_recorder.make_child('small_standard_sampling'), training_data, test_data, data_rng=data_rng, model_rng=model_rng)
                
                training_data, test_data = dataset.balanced_sampling(test_n=uniques_test_n, train_n=uniques_train_n, rng=data_rng)
                self.fit_and_evaluate_models(bootstrap_recorder.make_child('small_balanced_sampling'), training_data, test_data, data_rng=data_rng, model_rng=model_rng)

    def fit_and_evaluate_models(self, experiment_recorder: ExperimentRecorder, training_data: Dataset, test_data: Dataset, data_rng: np.random.Generator, model_rng: np.random.Generator):
        training_data.make_preprocessor()
        test_data.set_preprocessor(training_data.preprocessor)

        for model_conf in self.config.model_configs:
            model = Model(model_conf, model_rng)
            model_recorder = experiment_recorder.make_child(model.name)
            model.fit(training_data, model_recorder, data_rng)
            model.evaluate(test_data, model_recorder, data_rng)
            