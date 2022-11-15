from yieldprediction.dataset import DatasetConfig
from pathlib import Path
config = DatasetConfig(name='USPTO_Lowe', 
path=Path('uspto_lowe_rdkit_descriptors.ftr'),
format='feather',
yield_percentage_col=1,
c1_smiles_col=12, c2_smiles_col=13, p_smiles_col=14, 
c1_id_col=9, c2_id_col=10, p_id_col=11, 
feature_cols=slice(15, None))