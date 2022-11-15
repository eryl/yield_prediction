import argparse
from pathlib import Path
from collections import defaultdict
import json 

import numpy as np
import pandas as pd

from rdkit.Chem import MolToSmiles, MolFromSmiles
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from yieldprediction.utils import get_module_obj
from yieldprediction.dataset import DatasetConfig

def main():
    parser = argparse.ArgumentParser(description="Script for preprocessing the reaction data")
    parser.add_argument('dataset_config', help="Path to the dataset config to use", type=Path)
    parser.add_argument('--negative-threshold', help="Any yields under this threshold will be considered negative", type=int, default=40)
    parser.add_argument('--positive-threshold', help="Any yields above this threshold will be considered positive", type=int, default=60)
    parser.add_argument('--format', help='What format to write the data to. CSV is good for legacy, feather is much faster', choices=('csv', 'feather'), default='csv')

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
        
    feature_col = config.feature_cols.start
    if config.yield_percentage_col < feature_col:
        data.insert(loc=feature_col, column='product_id', value=p_ids)
        data.insert(loc=feature_col, column='component2_id', value=c2_ids)
        data.insert(loc=feature_col, column='component1_id', value=c1_ids)
        data.insert(loc=config.yield_percentage_col+1, column='yield_class', value=yield_class)
    else:
        data.insert(loc=config.yield_percentage_col+1, column='yield_class', value=yield_class)
        data.insert(loc=feature_col, column='product_id', value=p_ids)
        data.insert(loc=feature_col, column='component2_id', value=c2_ids)
        data.insert(loc=feature_col, column='component1_id', value=c1_ids)


    if args.format == 'csv':
        data_name = dataset_path.parent / f'{config.name}_with_duplicates-le{args.negative_threshold}-ge{args.positive_threshold}.tsv'
        data.to_csv(data_name, sep='\t', index=False)
    elif args.format == 'feather':
        data_name = dataset_path.parent / f'{config.name}_with_duplicates-le{args.negative_threshold}-ge{args.positive_threshold}.ftr'
        data = data.reset_index(drop=True)
        data.to_feather(data_name)

    duplicate_reactions = defaultdict(list)

    for i, reaction in enumerate(reaction_ids_unordered_c):
        duplicate_reactions[reaction].append(i)
    
    reactions_to_keep = []
    for reaction, duplicates in duplicate_reactions.items():
        by_yield = sorted(duplicates, key=lambda i: yield_percent[i])
        best_yield_index = by_yield[-1]
        reactions_to_keep.append(best_yield_index)

    filtered_data = data.iloc[sorted(reactions_to_keep)]

    if args.format == 'csv':
        filtered_data_name = dataset_path.parent / f'{config.name}_without_duplicates-le{args.negative_threshold}-ge{args.positive_threshold}.tsv'
        filtered_data.to_csv(filtered_data_name, sep='\t', index=False)
    elif args.format == 'feather':
        filtered_data_name = dataset_path.parent / f'{config.name}_without_duplicates-le{args.negative_threshold}-ge{args.positive_threshold}.ftr'
        filtered_data = filtered_data.reset_index(drop=True)
        filtered_data.to_feather(filtered_data_name)

    with open(dataset_path.parent / 'processed_columns.json', 'w') as fp:
        json.dump(list(enumerate(data.columns)), fp, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()


        
    
    
