# Training data directory
TRAIN_DIR: str = "/Volumes/NO_NAME/model_input_dir"
# Number of training epochs
DEFAULT_EPOCHS: int = 20
# Number of images per training batch
DEFAULT_BATCH_SIZE: int = 16
# Starting learning rate
DEFAULT_LEARNING_RATE: float = 0.001
# Penalty term applied to loss function to prevent overfitting
DEFAULT_WEIGHT_DECAY: float = 0.0005
# Number of epochs before reducing learning rate by gamma
DEFAULT_STEP_SIZE: int = 3
# Factor to decrease learning rate
DEFAULT_GAMMA: float = 0.1
# Train only classification and RPN
TRAIN_ONLY_TOP_LAYER: bool = True
# Number of epochs with increase val loss before stop training
EARLY_STOP_PATIENCE: int = 0
# Fraction of data to set aside for validation
VALIDATION_SPLIT: float = 0.20
