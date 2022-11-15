from yieldprediction.dataset import DatasetConfig
from pathlib import Path
config = DatasetConfig(name='buchwald-hartwig-nodupes', 
path=Path('buchwald-hartwig_without_duplicates-le40-ge60.ftr'),
format='feather',
yield_percentage_col=1, yield_class_col=2,
c1_smiles_col=7, c2_smiles_col=8, p_smiles_col=9,
c1_id_col=10, c2_id_col=11, p_id_col=12,
#separator='\t', 
feature_cols=slice(13, None))