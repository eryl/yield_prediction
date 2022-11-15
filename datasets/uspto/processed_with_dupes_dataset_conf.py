from yieldprediction.dataset import DatasetConfig
from pathlib import Path
config = DatasetConfig(name='USPTO_Lowe', 
path=Path('USPTO_Lowe_with_duplicates-le40-ge60.ftr'),
format='feather',
yield_percentage_col=1, yield_class_col=2,
c1_smiles_col=13, c2_smiles_col=14, p_smiles_col=15, 
c1_id_col=10, c2_id_col=11, p_id_col=12, 
feature_cols=slice(19, None))