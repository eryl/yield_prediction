from yieldprediction.dataset import DatasetConfig
from pathlib import Path
config = DatasetConfig(name='suzuki', 
path=Path('Suzuki_all_non_filtered_RDKIT.tsv'),
c1_smiles_col=7, c2_smiles_col=8, p_smiles_col=9, yield_percentage_col=1, separator='\t', feature_cols=slice(10, None))