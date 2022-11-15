from yieldprediction.models.model import ModelConfig
from yieldprediction.models.learningalgorithms.identy_based_catboost import CatBoostConfig, IdentityBasedCatBoost

catboost_config = CatBoostConfig()

model_config = ModelConfig(model_type=IdentityBasedCatBoost, model_kwargs=dict(config=catboost_config))