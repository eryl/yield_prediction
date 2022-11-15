

from dataclasses import dataclass

from catboost import CatBoostClassifier, Pool, metrics, cv

from yieldprediction.experiment.experiment_recorder import ExperimentRecorder
from yieldprediction.models.learningalgorithms.identity_based import IdentityBasedLearningAlgorithm
from yieldprediction.dataset import Dataset

@dataclass
class CatBoostConfig:
    iterations: int = 5000
    learning_rate: float = 1e-3
    fold_count: int = 5
    dev_ratio: float = 0.1

class IdentityBasedCatBoost(IdentityBasedLearningAlgorithm):
    def __init__(self, *, config: CatBoostConfig, dataset: Dataset = None, rng=None) -> None:
        super().__init__(dataset=dataset, rng=rng)
        self.config = config
        cat_features = [0,1,2]  # All features are categorical
        self.model = CatBoostClassifier(iterations=config.iterations, cat_features=cat_features,  learning_rate=config.learning_rate, random_state=self.rng.integers(2**31 - 1))

    def fit(self, training_dataset: Dataset, experiment_recorder: ExperimentRecorder, dev_dataset: Dataset = None, rng=None):
        train_split, dev_split = training_dataset.standard_sampling(self.config.dev_ratio, rng=rng)
        X_train = train_split.get_id_features()
        y_train = train_split.get_labels()

        X_dev = dev_split.get_id_features()
        y_dev = dev_split.get_labels()

        self.model.fit(X_train, y_train, eval_set=(X_dev, y_dev))

    def predict_proba(self, dataset: Dataset):
        X = dataset.get_id_features()
        predictions = self.model.predict_proba(X)
        pos_class = list(self.model.classes_).index(1)
        return predictions[:,pos_class]