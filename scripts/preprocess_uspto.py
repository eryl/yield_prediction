import argparse
from pathlib import Path
import csv

import pandas as pd

from rdkit.Chem import MolToSmiles, AllChem
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Script for preprocessing the USPTO data, removing salts and keeping reactions with two reactants")
    parser.add_argument('csv_files', help="Path to the CSV file to process", type=Path, nargs='+')
    parser.add_argument('output_file', help="Path to the CSV file to process", type=Path)
    parser.add_argument('--threshold', help="Treshold to use when filtering the unmapped reactants", type=float, default=0.2)
    args = parser.parse_args()


    seen_reactions = 0
    molecules = set()
    kept_reactions = list()

    for csv_file in args.csv_files:
        csv_basename = csv_file.with_suffix('').name
        data = pd.read_csv(csv_file, sep='\t', index_col=None)
        if 'ReactionSmiles' in data.columns:
            reaction_column = 'ReactionSmiles'
        elif 'OriginalReaction' in data.columns:
            reaction_column = 'OriginalReaction'
        else:
            raise ValueError("No columns with mapped reaction found")
        # We first go through all reactions and process the constituents
        for i, row in tqdm(data.iterrows(), desc='Processing SMARTS', total=len(data)):
            smarts = row[reaction_column]
            try:
                reaction = AllChem.ReactionFromSmarts(smarts)
            except RuntimeError as e:
                print(e)
                continue
            reaction.RemoveUnmappedReactantTemplates(args.threshold)
            reaction.RemoveUnmappedProductTemplates(args.threshold)
            if reaction.GetNumReactantTemplates() == 2 and reaction.GetNumProductTemplates() == 1:
                AllChem.RemoveMappingNumbersFromReactions(reaction)
                reactant_1, reactant_2 = reaction.GetReactants()
                product, *_ = reaction.GetProducts()
                component_1_smiles = MolToSmiles(reactant_1, canonical=True)
                component_2_smiles = MolToSmiles(reactant_2, canonical=True)
                product_smiles = MolToSmiles(product, canonical=True)

                molecules.update({component_1_smiles, component_2_smiles, product_smiles})
                if 'myID' in row:
                    my_id = row['myID']
                else:
                    my_id = seen_reactions
                # TODO: Figure out if we can populate Catalyst and Solvent automatically
                kept_reactions.append({"DataSetName": csv_basename, 
                                        "yield_percent": row['Yield'], 
                                        "Reaction ID": i, 
                                        "Catalyst": 'None', 
                                        "Solvent": 'None', 
                                        "my_ID": my_id, 
                                        "component1_(smiles)": component_1_smiles, 
                                        "component2_(smiles)": component_2_smiles, 
                                        "product_(smiles)": product_smiles})
                seen_reactions += 1

    # TODO: If we want some other way of ordering components we can do it by the ordering assigned here
    molecules_ids = {mol: i for i, mol in enumerate(sorted(molecules))}
    
    # This block writes out the results
    fieldnames = ["DataSetName", "yield_percent", "Reaction ID", "Catalyst", "Solvent", "my_ID", "component1_(smiles)", "component2_(smiles)", "product_(smiles)"]
    
    with open(args.output_file, 'w', newline='') as fp:
        csv_writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter='\t')
        csv_writer.writeheader()
        for reaction in tqdm(kept_reactions, desc='Writing reactions'):
            comp_1_id = molecules_ids[reaction['component1_(smiles)']]
            comp_2_id = molecules_ids[reaction['component2_(smiles)']]
            if comp_1_id > comp_2_id:
                reaction['component2_(smiles)'], reaction['component1_(smiles)'] = reaction['component1_(smiles)'], reaction['component2_(smiles)']
            csv_writer.writerow(reaction)
    


if __name__ == '__main__':
    main()


        
    
    
