from argparse import ArgumentError
from dataclasses import dataclass
from functools import total_ordering
from os import access
from pathlib import Path
from numbers import Integral
from collections.abc import Sequence
from re import X
from typing import List, Union, Tuple, Optional
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.impute import SimpleImputer

from yieldprediction.utils import get_module_obj

@dataclass
class DatasetConfig:
    name: str
    path: Path
    c1_smiles_col: int
    c2_smiles_col: int
    p_smiles_col: int
    yield_percentage_col: int
    feature_cols: Union[slice, List[int]]
    separator: str = '\t'
    yield_class_col: Optional[int] = None
    c1_id_col: Optional[int] = None
    c2_id_col: Optional[int] = None
    p_id_col: Optional[int] = None
    format: str = 'csv'


class Dataset:
    @classmethod
    def from_config(cls, config_path: Path):
        config = get_module_obj(config_path, DatasetConfig)
        dataset_path = config_path.parent / config.path

        if config.format == 'csv':
            data = pd.read_csv(dataset_path, sep=config.separator, index_col=None)
        elif config.format == 'feather':
            data = pd.read_feather(dataset_path)
        
        reaction_triplets = data.iloc[:, [config.c1_id_col, config.c2_id_col, config.p_id_col]].values
        yield_class = data.iloc[:, config.yield_class_col].values
        yield_percent = data.iloc[:, config.yield_percentage_col].values
        features = data.iloc[:, config.feature_cols].values
        
        return Dataset(config=config, reaction_triplets=reaction_triplets, rdkit_featuers=features, yield_percent=yield_percent, yield_class=yield_class)

    def __init__(self, *, config, reaction_triplets, rdkit_featuers, yield_percent, yield_class, index_subset: List[int] = None) -> None:
        self.config = config
        self.name = self.config.name
        self.reaction_triplets = reaction_triplets
        self.rdkit_features = rdkit_featuers
        
        self.yield_percent = yield_percent
        self.yield_class = yield_class

        if index_subset is None:
            index_subset = list(range(len(yield_class)))

        self.index_subset = sorted(index_subset)
        #self.inv_index_map = {ri:vi for vi, ri in enumerate(self.index_subset)}  # We want to be able to make an inverse lookup, we use a mapping from real indices to "virtual" ones
        
        self.label_indices = defaultdict(list)
        
        # Since the index subset is ordered, so will the label conditioned indices. 
        # We set the label indices to actually use the virtual indices to make outside 
        # algorithms oblivious to use keeping the full dataset behind the scenes
        for vi, ri in enumerate(self.index_subset):
            y = self.yield_class[ri]
            self.label_indices[y].append(vi)

        self.num_molecule_ids = self.reaction_triplets.max() + 1  #Plus one due to zero indexing

        self.preprocessor = None

    def __len__(self):
        return len(self.index_subset)

    def virtual_to_real_index(self, virtual_index=None):
        if virtual_index is None:
            virtual_index = slice(None)

        if isinstance(virtual_index, Integral):
            real_index = self.index_subset[virtual_index]
            return real_index
        else:
            if isinstance(virtual_index, (list, np.ndarray)):
                real_indices = [self.index_subset[vi] for vi in virtual_index]
            elif isinstance(virtual_index, slice):
                real_indices = self.index_subset[virtual_index]
            else:
                raise NotImplementedError(f'__getitem__ for {virtual_index} of type {type(virtual_index)} has not been implemented')
            return real_indices

    def make_preprocessor(self):
        features = self.get_rdkit_features()

        imputer = SimpleImputer()
        features = imputer.fit_transform(features, None)
        scaler = MinMaxScaler(clip=True)
        scaler.fit(features)
        self.preprocessor = make_pipeline(imputer, scaler)

    def set_preprocessor(self, preprocessor: Pipeline):
        self.preprocessor = preprocessor

    def get_rdkit_features(self, virtual_index=None, preprocessor=None):
        if preprocessor is None:
            preprocessor = self.preprocessor
        
        features = self.rdkit_features[self.virtual_to_real_index(virtual_index)]

        if preprocessor is not None:
            features = preprocessor.transform(features)
        
        return features

    def get_id_features(self, virtual_index=None):
        return self.reaction_triplets[self.virtual_to_real_index(virtual_index)]

    def get_labels(self, virtual_index=None):
        return self.yield_class[self.virtual_to_real_index(virtual_index)]

    def stratfied_label_index_sampling(self, label_indices, test_ratio=None, train_n=None, test_n=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        shuffled_label_indices = {label: rng.permuted(label_indices) for label, label_indices in label_indices.items()}
        total_n = sum(len(indices) for indices in shuffled_label_indices.values())
        

        # There's essentially three important combinations of assignment to test_ratio, train_n and test_n
        # If test_ratio is set, but train_n and test_n is not, then the whole dataset is split into train_n = n*(1-test_ratio), test_n = n*test_ratio
        # If test_ratio and train_n is set, but not test_n, then test_n is set such that it would be test_ratio*virtual_n of a virtual dataset where test_n=virtual_n*(1-test_ratio)
        # If train_n and test_n is both set, then they are used as is
        
        assert test_ratio is not None or train_n is not None or test_n is not None, "At least one of test_ratio, train_n or test_n has to be assigned a value"
        

        if train_n is None:
            if test_n is None:
                test_n = int(math.ceil(total_n * test_ratio))
                train_n = total_n - test_n
            else:
                # test_n is set. Is assumed to be test_ratio of a full dataset of size x, where the size is unknown. 
                # train_n can then be found by solving for
                # test_n = test_ratio*x, train_n = (1-test_ratio)*x
                # train_n/(1-test_ratio) = x, substitute x:
                # test_n = test_ratio * train_n/(1-test_ratio)
                # train_n = test_n * (1-test_ratio)/test_ratio
                train_n = test_n * (1-test_ratio)/test_ratio
                
        else:
            if test_n is None:
                # train_n is set. It's assumed to be 1-test_ratio of a full dataset of size x, where the size is unknown. 
                # train_n can then be found by solving for
                # train_n = (1-test_ratio)*x, test_n = test_ratio*x
                # test_n/test_ratio = x, substitute x:
                # train_n = (1-test_ratio)*test_n/test_ratio
                # test_n = train_n * test_ratio/(1-test_ratio)
                test_n = train_n * test_ratio/(1-test_ratio)
                
        total_n = test_n + train_n
        training_ratio = train_n / total_n
        test_ratio = 1 - training_ratio  # We reset the test ratio here

        training_indices = dict()
        test_indices = dict()
        
        for label, indices in shuffled_label_indices.items():
            label_n = len(indices)
            n_train_indices = int(math.floor(label_n * training_ratio))
            n_test_indices = int(math.ceil(label_n * test_ratio))
            assert n_train_indices + n_test_indices <= label_n, "The test and train indices will overlap, somethings awry"
            training_indices[label] = indices[:n_train_indices]
            test_indices[label] = indices[-n_test_indices:]
            if set(training_indices[label]).intersection(test_indices[label]):
                raise RuntimeError("Training and test dataset overlap")

        return training_indices, test_indices
        
        # Now subsample based on test_ratio. If test_from_dropped and n was not None, we sample the test set from the excluded examples
        
    def standard_sampling(self, test_ratio=None, train_n=None, test_n=None, rng=None):
        """Does standard stratified sampling of the dataset. Returns two new Dataset objects. 
        :param r: Ratio of dataset to sample. If n is also"""
        if rng is None:
            rng = np.random.default_rng()

        train_labeled_virtual_indices, test_labeled_virtual_indices = self.stratfied_label_index_sampling(self.label_indices, test_ratio=test_ratio, train_n=train_n, test_n=test_n, rng=rng)
        # Convert the vritual indices to a real index subset
        train_index_subset = [self.index_subset[index] for indices in train_labeled_virtual_indices.values() for index in indices]
        test_index_subset = [self.index_subset[index] for indices in test_labeled_virtual_indices.values() for index in indices]

        train_dataset = Dataset(config=self.config, reaction_triplets=self.reaction_triplets, rdkit_featuers=self.rdkit_features, yield_percent=self.yield_percent, yield_class=self.yield_class, index_subset=train_index_subset)
        test_dataset = Dataset(config=self.config, reaction_triplets=self.reaction_triplets, rdkit_featuers=self.rdkit_features, yield_percent=self.yield_percent, yield_class=self.yield_class, index_subset=test_index_subset)
        return train_dataset, test_dataset
        

    def balanced_sampling(self, test_ratio=None, train_n=None, test_n=None, rng=None):
        """Does stratified sampling but undersamples the majority class to achieve balanced classes"""
        if rng is None:
            rng = np.random.default_rng()

        # Find the count of the minority class
        minority_n = min(len(label_examples) for label_examples in self.label_indices.values())

        # Downsample all classes to be of equal size to the minority class. We start by shuffling the indices for the labels.
        # We use rng.permuted since its not in place in difference from rng.shuffle
        shuffled_label_indices = {label: rng.permuted(label_indices) for label, label_indices in self.label_indices.items()}
        # Now create a new label_indices by slicing out the minority_n first elements
        shuffled_label_indices = {label: indices[:minority_n] for label, indices in shuffled_label_indices.items()}
        
        train_labeled_virtual_indices, test_labeled_virtual_indices = self.stratfied_label_index_sampling(shuffled_label_indices, test_ratio=test_ratio, train_n=train_n, test_n=test_n, rng=rng)
        # Convert the vritual indices to a real index subset
        train_index_subset = [self.index_subset[index] for indices in train_labeled_virtual_indices.values() for index in indices]
        test_index_subset = [self.index_subset[index] for indices in test_labeled_virtual_indices.values() for index in indices]

        train_dataset = Dataset(config=self.config, reaction_triplets=self.reaction_triplets, rdkit_featuers=self.rdkit_features, yield_percent=self.yield_percent, yield_class=self.yield_class, index_subset=train_index_subset)
        test_dataset = Dataset(config=self.config, reaction_triplets=self.reaction_triplets, rdkit_featuers=self.rdkit_features, yield_percent=self.yield_percent, yield_class=self.yield_class, index_subset=test_index_subset)
        return train_dataset, test_dataset

        
    def uniques_sampling(self, test_ratio=None, train_n=None, test_n=None, rng=None):
        """Returns a random subset of the smiles_triplets data such that only unique reactions will be present."""
        if rng is None:
            rng = np.random.default_rng()

        shuffled_indices = rng.permutation(len(self))
        used_molecules = set()
        selected_train_indices = list()
        unselected_label_indices = defaultdict(list)

        for virtual_index in shuffled_indices:
            real_index = self.index_subset[virtual_index]
            molecules = set(self.reaction_triplets[real_index])
            if not used_molecules.intersection(molecules):
                used_molecules.update(molecules)
                selected_train_indices.append(real_index)
            else:
                label = self.yield_class[real_index]
                unselected_label_indices[label].append(real_index)

        train_n = len(selected_train_indices)
        test_n = int(round((test_ratio/(1-test_ratio)) * train_n))  # Take the test dataset as if the training dataset was subsample of ratio (1-test_ratio) from a dataset 

        # The *unselected_label_indices* are shuffled, since it was constructed from a shuffled list.
        unselected_n = sum(len(indices) for indices in unselected_label_indices.values())
        label_ratios = {label: len(indices)/unselected_n for label, indices in unselected_label_indices.items()}
        sorted_ratios = sorted(label_ratios.items(), key=lambda x: x[1])
        
        # This convoluted code is just to make sure we get the right amount samples, and that we round the fractions in favor of the  minority class (if the ratios don't add upp to an even number of example, the odd one is added to the minority class)
        label_ns = {label: int(math.floor(test_n*label_ratio)) for label, label_ratio in sorted_ratios[1:]}
        non_minority_n = sum(label_ns.values())
        minority_label, minority_ratio = sorted_ratios[0]
        label_ns[minority_label] = test_n - non_minority_n
        
        selected_test_indices = list()
        for label, label_n in label_ns.items():
            selected_test_indices.extend(unselected_label_indices[label][:label_n])
        
        train_dataset = Dataset(config=self.config, reaction_triplets=self.reaction_triplets, rdkit_featuers=self.rdkit_features, yield_percent=self.yield_percent, yield_class=self.yield_class, index_subset=selected_train_indices)
        test_dataset = Dataset(config=self.config, reaction_triplets=self.reaction_triplets, rdkit_featuers=self.rdkit_features, yield_percent=self.yield_percent, yield_class=self.yield_class, index_subset=selected_test_indices)
        return train_dataset, test_dataset









        
        
