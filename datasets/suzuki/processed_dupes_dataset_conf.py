from yieldprediction.dataset import DatasetConfig
from pathlib import Path
config = DatasetConfig(name='suzuki-dupes', 
path=Path('suzuki_with_duplicates-le40-ge60.ftr'),
format='feather',
yield_percentage_col=1, yield_class_col=2,
c1_smiles_col=8, c2_smiles_col=9, p_smiles_col=10,
c1_id_col=11, c2_id_col=12, p_id_col=13,
#separator='\t', 
feature_cols=slice(14, None))