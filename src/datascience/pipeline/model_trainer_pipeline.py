from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience import logger 

STAGE_NAME = 'MODEL TRAINING'

class ModelTrainerPipeline:
    def __init__(self):
        pass
    
    def _initiate_model_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_conf()
        model_trainer = ModelTrainer(config = model_trainer_config)
        model_trainer.train()