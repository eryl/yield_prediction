from abc import ABC

import numpy as np 

from yieldprediction.dataset import Dataset
from yieldprediction.experiment.experiment_recorder import ExperimentRecorder

class LearningAlgorithm(ABC):
    def __init__(self, *, dataset: Dataset=None, rng: np.random.Generator=None) -> None:
        super().__init__()
        self.dataset = dataset

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
    
    def fit(self, training_dataset: Dataset, experiment_recorded: ExperimentRecorder, rng=None):
        if rng is None:
            rng = np.random.default_rng()

    def evaluate(self, test_dataset: Dataset, experiment_recorder: ExperimentRecorder):
        pass

    def predict_proba(self, dataset: Dataset):
        raise NotImplementedError("predict_proba has not been implemented")
