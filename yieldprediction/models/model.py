from dataclasses import dataclass, field
from distutils.command.config import config
from typing import List, Optional, Any, Dict
from pathlib import Path

import numpy as np

from yieldprediction.utils import get_module_obj
from yieldprediction.dataset import Dataset
from yieldprediction.models.learningalgorithms.base_algorithm import LearningAlgorithm
from yieldprediction.experiment.experiment_recorder import ExperimentRecorder

@dataclass
class ModelConfig:
    model_type: type[LearningAlgorithm]
    model_args: List[Any] = field(default_factory=tuple)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ''

class Model:
    def __init__(self, config_path: Path, rng: np.random.Generator) -> None:
        self.config = get_module_obj(config_path, ModelConfig)
        self.name = self.config.model_type.__name__
        self.rng = rng

    def fit(self, training_dataset: Dataset, experiment_recorder: ExperimentRecorder, data_rng: np.random.Generator=None):
        self.model = self.config.model_type(*self.config.model_args, **self.config.model_kwargs, dataset=training_dataset, rng=self.rng)
        self.model.fit(training_dataset, experiment_recorder, rng=data_rng)

    def evaluate(self, test_data: Dataset, experiment_recorder: ExperimentRecorder, data_rng: np.random.Generator=None):
        if not hasattr(self, 'model'):
            raise RuntimeError("Model.evaluate() called before fit(). Please fit the model before running evaluation.")
        predictions = self.model.predict_proba(test_data)
        targets = test_data.get_labels()
        experiment_recorder.record_performance(predictions, targets, tag='test')
    

