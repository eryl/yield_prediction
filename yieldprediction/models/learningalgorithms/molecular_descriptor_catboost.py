

from dataclasses import dataclass

from catboost import CatBoostClassifier, Pool, metrics, cv
import numpy as np

from yieldprediction.experiment.experiment_recorder import ExperimentRecorder
from yieldprediction.models.learningalgorithms.molecular_descriptor_based import MolecularDescriptorsLearningAlgorithm
from yieldprediction.dataset import Dataset

@dataclass
class CatBoostConfig:
    iterations: int = 500
    learning_rate: float = 1e-4
    fold_count: int = 5

class MolecularDescriptorCatBoost(MolecularDescriptorsLearningAlgorithm):
    def __init__(self, *, config: CatBoostConfig, rng: np.random.Generator, dataset: Dataset = None, ) -> None:
        super().__init__(dataset=dataset, rng=rng)
        self.config = config
        self.model = CatBoostClassifier(iterations=config.iterations, learning_rate=config.learning_rate, random_state=self.rng.integers(2**31 - 1))

    def fit(self, training_dataset: Dataset, experiment_recorder: ExperimentRecorder, rng=None):
        
        X = training_dataset.get_rdkit_features()
        y = training_dataset.get_labels()
        #cv_dataset = Pool(X, y)
        #cv_params = dict(iterations=self.config.iterations, learning_rate=self.config.learning_rate, loss_function=metrics.Logloss())
        #cv_data = cv(cv_dataset, params=cv_params, plot=False, stratified=True, fold_count=self.config.fold_count, return_models=True, early_stopping_rounds=20)

        self.model.fit(X, y)

    def predict_proba(self, dataset: Dataset):
        X = dataset.get_rdkit_features()
        predictions = self.model.predict_proba(X)
        pos_class = list(self.model.classes_).index(1)
        return predictions[:,pos_class]