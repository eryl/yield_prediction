import argparse
from pathlib import Path

from yieldprediction.experiment import Experiment
from yieldprediction.utils import timestamp

def main():
    parser = argparse.ArgumentParser(description="Script for running the yield prediction experiments")
    parser.add_argument('config_file', help="Path to the experiment configuration file", type=Path)
    parser.add_argument('--experiment-root', help="Root directory for the recorded experiment data. A timestamp will be appended to this path.", type=Path, default='experiments')
    
    args = parser.parse_args()

    experiment = Experiment(args.experiment_root / timestamp(), args.config_file)
    experiment.run_experiments()
    

if __name__ == '__main__':
    main()


