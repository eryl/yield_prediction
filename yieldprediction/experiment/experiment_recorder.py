from pathlib import Path

from sklearn.metrics import roc_auc_score

class ExperimentRecorder:
    def __init__(self, recorder_path: Path) -> None:
         self.path = recorder_path
         self.path.mkdir(parents=True)

    def make_child(self, child_name):
        return ExperimentRecorder(self.path / child_name)

    def record_performance(self, predictions, targets, tag=None):
        roc_auc = roc_auc_score(targets, predictions)
        with open(self.path / f'{tag}_performance.txt', 'w') as fp:
            fp.write(f'ROC_AUC: {roc_auc}')