import argparse
import itertools
from pathlib import Path
from collections import Counter, defaultdict
import json 

import numpy as np
import pandas as pd
import networkx

from rdkit.Chem import MolToSmiles, MolFromSmiles
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from yieldprediction.utils import get_module_obj
from yieldprediction.dataset import DatasetConfig

def main():
    parser = argparse.ArgumentParser(description="Script for creating graphs based on the reaction data")
    parser.add_argument('dataset_config', help="Path to the dataset config to use", type=Path)
    parser.add_argument('--format', help='What format to write the data to. edgelist outputs a comma-separated list of edges and weights, json outputs a adjacency list as a json object', choices=('edgelist', 'json'), default='csv')

    args = parser.parse_args()

    config = get_module_obj(args.dataset_config, DatasetConfig)

    dataset_path = args.dataset_config.parent / config.path
    dataset_format = config.format
    if dataset_format == 'csv':
        data = pd.read_csv(dataset_path, sep=config.separator, index_col=None)
    elif dataset_format == 'feather':
        data = pd.read_feather(dataset_path)
    else:
        raise NotImplementedError(f"Format {format} is not supported")
    yield_percent = data.iloc[:, config.yield_percentage_col].values
    
    data = data[np.logical_or(yield_percent <= args.negative_threshold, yield_percent >= args.positive_threshold)]
    yield_percent = data.iloc[:, config.yield_percentage_col].values
    
    # Drop inf values from the dataframe
    data = data.replace([np.inf, -np.inf], np.nan)
    
    max_i = 0
    molecule_ids = dict()
    inv_molecule_ids = dict()
    c1_ids = []
    c2_ids = []
    p_ids = []

    reaction_ids = []
    reaction_ids_unordered_c = []
    
    if config.c1_id_col is None or config.c2_id_col is None or config.p_id_col is None:
        reaction_triplets = data.iloc[:, [config.c1_smiles_col, config.c2_smiles_col, config.p_smiles_col]].values
        for c1, c2, p in tqdm(reaction_triplets, desc="Canonicalizing molecules"):
            canonical_c1 = MolToSmiles(MolFromSmiles(c1), canonical=True)
            canonical_c2 = MolToSmiles(MolFromSmiles(c2), canonical=True)
            canonical_p = MolToSmiles(MolFromSmiles(p), canonical=True)
            for smiles in (canonical_c1, canonical_c2, canonical_p):
                if smiles not in molecule_ids:
                    molecule_ids[smiles] = max_i
                    inv_molecule_ids[max_i] = smiles
                    max_i += 1
            id_c1 = molecule_ids[canonical_c1]
            id_c2 = molecule_ids[canonical_c2]
            id_p = molecule_ids[canonical_p]
            c1_ids.append(id_c1)
            c2_ids.append(id_c2)
            p_ids.append(id_p)

            reaction_ids.append((id_c1, id_c2, id_p))
            reaction_ids_unordered_c.append((frozenset({id_c1, id_c2}), id_p))
    else:
        columns = [config.c1_id_col, config.c2_id_col, config.p_id_col]
        reaction_id_triplets = data.iloc[:, columns].values
        for id_c1, id_c2, id_p in tqdm(reaction_id_triplets, desc="Collating reactions"):
            c1_ids.append(id_c1)
            c2_ids.append(id_c2)
            p_ids.append(id_p)

            reaction_ids.append((id_c1, id_c2, id_p))
            reaction_ids_unordered_c.append((frozenset({id_c1, id_c2}), id_p))

    yield_class = np.array([0 if x <= args.negative_threshold else 1 if x >= args.positive_threshold else float('nan') for x in yield_percent])
    gaussian_yield = []  # We will use the percentage values to calculate (using a Bayesian update rule) the yield distribution per molecules
    beta_yield = []  # We will use the yield class to calculate (using a Bayesian update rule) the yield class distribution per molecules
    
    # Here we build the graph
    reactions_graph = defaultdict(Counter)
    for reaction_triplet in reaction_ids:
        for m1, m2 in itertools.combinations(reaction_id_triplets, 2):
            reactions_graph[frozenset(m1, m2)] += 1
    
    
    if args.format == 'json':
        ...
    elif args.format == 'edgelist':
        ...

    
if __name__ == '__main__':
    main()


        
    
    
