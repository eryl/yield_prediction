from yieldprediction.dataset import DatasetConfig
from pathlib import Path
config = DatasetConfig(name='buchwald-hartwig', 
path=Path('BuchwaldHartwig_all_non_filtered_RDKIT.tsv'),
c1_smiles_col=6, c2_smiles_col=7, p_smiles_col=8, yield_percentage_col=1, separator='\t', feature_cols=slice(9, None))