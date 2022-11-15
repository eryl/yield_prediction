from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from yieldprediction.experiment.experiment_recorder import ExperimentRecorder

from yieldprediction.models.learningalgorithms.molecular_descriptor_based import MolecularDescriptorsLearningAlgorithm
from yieldprediction.dataset import Dataset

@dataclass
class LogisticRegressionConfig:
    penalty: str = 'l2'
    C: float = 1.
    class_weight: str = 'balanced'

class MolecularDescriptorLogisticRegression(MolecularDescriptorsLearningAlgorithm):
    def __init__(self, *, config: LogisticRegressionConfig, dataset: Dataset = None) -> None:
        super().__init__(dataset=dataset)
        self.model = LogisticRegression(penalty=config.penalty, C=config.C, class_weight=config.class_weight)

    def fit(self, training_dataset: Dataset, experiment_recorder: ExperimentRecorder, rng=None):
        X = training_dataset.get_rdkit_features()
        y = training_dataset.get_labels()

        self.model.fit(X, y)

    def predict_proba(self, dataset: Dataset):
        X = dataset.get_rdkit_features()
        predictions = self.model.predict_proba(X)
        pos_class = list(self.model.classes_).index(1)
        return predictions[:,pos_class]
    
    