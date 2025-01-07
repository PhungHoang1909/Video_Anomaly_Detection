import torch
from pathlib import Path

class Config:
    # Paths
    DATA_PATH = Path('/kaggle/input/mem-vad/MEM_Video-Anomaly-Detection/datasets/ped2')
    CHECKPOINT_DIR = Path('/kaggle/working')

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data parameters
    INPUT_SIZE = (128, 128)
    SEQUENCE_LENGTH = 8
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    
    # Model parameters
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    EMBED_DIM = 256
    DEPTH = 4
    NUM_HEADS = 8
    MLP_RATIO = 4.0
    LOCAL_SIZE = 8
    SPARSITY = 0.5
    
    # Training parameters
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    SCHEDULER_GAMMA = 0.95
    SCHEDULER_STEP_SIZE = 5
    GRAD_CLIP_NORM = 1.0
    
    # Early stopping
    PATIENCE = 10
    DELTA = 1e-6
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    
    # Loss weights
    RECON_LOSS_WEIGHT = 1.0
    FEATURE_LOSS_WEIGHT = 0.5
    
    @staticmethod
    def make_dirs():
        """Create necessary directories."""
        Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_config(filename):
        """Save configuration to a file."""
        with open(filename, 'w') as f:
            for key, value in Config.__dict__.items():
                if not key.startswith('__') and not callable(value):
                    f.write(f'{key} = {value}\n')
