from yieldprediction.models.model import ModelConfig
from yieldprediction.models.learningalgorithms.molecular_descriptor_catboost import CatBoostConfig, MolecularDescriptorCatBoost

catboost_config = CatBoostConfig()

model_config = ModelConfig(model_type=MolecularDescriptorCatBoost, model_kwargs=dict(config=catboost_config))