import os
from pathlib import Path
import torch

class PathConfig:
    """Configuration for project paths."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"

    @classmethod
    def ensure_dirs_exist(cls):
        """Create necessary directories if they don't exist."""
        for attr_name in dir(cls):
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_value, Path) and not attr_name.startswith('_') and attr_name != 'BASE_DIR':
                attr_value.mkdir(parents=True, exist_ok=True)

class ModelConfig:
    """Configuration for the AetherManipulator model."""
    MODEL_NAME = "AetherManipulator_v1"
    INPUT_DIM = 768
    HIDDEN_DIMS = [512, 256]
    OUTPUT_DIM = 128
    ACTIVATION_FUNCTION = "relu"
    DROPOUT_RATE = 0.1
    USE_BATCH_NORM = True

class TrainingConfig:
    """Configuration for the training process."""
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    OPTIMIZER = "adam"  # options: "adam", "sgd", "rmsprop"
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10
    SCHEDULER_STEP_SIZE = 30
    SCHEDULER_GAMMA = 0.1
    GRADIENT_CLIP_VALUE = 1.0

class EnvironmentConfig:
    """Configuration for the execution environment."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4)) # Use environment variable or default
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Use environment variable or default

# Instantiate configurations
paths = PathConfig()
model_params = ModelConfig()
training_params = TrainingConfig()
env_config = EnvironmentConfig()

# Ensure directories exist upon import
paths.ensure_dirs_exist()

# You can optionally add a function to print or log the config
def log_config():
    import logging
    logging.basicConfig(level=env_config.LOG_LEVEL)
    logger = logging.getLogger(__name__)

    logger.info("--- Path Configuration ---")
    for key, value in vars(paths).items():
        if not key.startswith('_'):
             logger.info(f"{key}: {value}")

    logger.info("--- Model Configuration ---")
    for key, value in vars(model_params).items():
        if not key.startswith('_'):
            logger.info(f"{key}: {value}")

    logger.info("--- Training Configuration ---")
    for key, value in vars(training_params).items():
        if not key.startswith('_'):
            logger.info(f"{key}: {value}")

    logger.info("--- Environment Configuration ---")
    for key, value in vars(env_config).items():
        if not key.startswith('_'):
            logger.info(f"{key}: {value}")

if __name__ == "__main__":
    # Example of how to use the config
    print(f"Base Directory: {paths.BASE_DIR}")
    print(f"Data Directory: {paths.DATA_DIR}")
    print(f"Model Name: {model_params.MODEL_NAME}")
    print(f"Learning Rate: {training_params.LEARNING_RATE}")
    print(f"Device: {env_config.DEVICE}")
    print(f"Random Seed: {env_config.RANDOM_SEED}")

    # Log the configuration if run directly
    # log_config() # Uncomment to log config when script is run