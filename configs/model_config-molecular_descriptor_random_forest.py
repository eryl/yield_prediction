from yieldprediction.models.model import ModelConfig
from yieldprediction.models.learningalgorithms.random_forest import RandomForestPredictorConfig, RandomForestPredictor

rf_config = RandomForestPredictorConfig(n_estimators=300, n_workers=10)

model_config = ModelConfig(model_type=RandomForestPredictor, model_kwargs=dict(config=rf_config))