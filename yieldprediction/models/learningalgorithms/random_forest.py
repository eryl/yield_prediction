from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier

from yieldprediction.dataset import Dataset
from yieldprediction.experiment.experiment_recorder import ExperimentRecorder
from yieldprediction.models.learningalgorithms.molecular_descriptor_based import MolecularDescriptorsLearningAlgorithm

@dataclass
class RandomForestPredictorConfig:
    n_estimators: int
    n_workers: int


class RandomForestPredictor(MolecularDescriptorsLearningAlgorithm):
    def __init__(self, *, config: RandomForestPredictorConfig, dataset: Dataset = None, rng=None) -> None:
        super().__init__(dataset=dataset)
        self.model = RandomForestClassifier(n_estimators=config.n_estimators, n_jobs=config.n_workers)

    def fit(self, training_data: Dataset, experiment_recorder: ExperimentRecorder, rng=None):
        X = training_data.get_rdkit_features()
        y = training_data.get_labels()
        self.model.fit(X, y)

    def predict_proba(self, dataset: Dataset):
        X = dataset.get_rdkit_features()
        predictions = self.model.predict_proba(X)
        pos_class = list(self.model.classes_).index(1)
        return predictions[:,pos_class]