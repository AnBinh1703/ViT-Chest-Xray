"""
Configuration file for the ViT Chest X-Ray project.
This file sets up paths that work across different operating systems.
"""
import os

# Get the project root directory
def get_project_root():
    """Get the root directory of the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to project root (src/utils -> src -> root)
    return os.path.dirname(os.path.dirname(current_dir))

PROJECT_ROOT = get_project_root()

# Data paths
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
LABELS_CSV = os.path.join(DATA_ROOT, "Data_Entry_2017_v2020.csv")

# Model save directory
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Processed data directory
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Image settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 15

# Labels for classification
LABELS = [
    'Cardiomegaly', 'Emphysema', 'Effusion',
    'Hernia', 'Nodule', 'Pneumothorax', 'Atelectasis',
    'Pleural_Thickening', 'Mass', 'Edema', 'Consolidation',
    'Infiltration', 'Fibrosis', 'Pneumonia', 'No Finding'
]

# Training settings
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
NUM_EPOCHS = 10

def print_config():
    """Print the current configuration."""
    print("=" * 50)
    print("Project Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Images Directory: {IMAGES_DIR}")
    print(f"Labels CSV: {LABELS_CSV}")
    print(f"Checkpoints Directory: {CHECKPOINTS_DIR}")
    print(f"Processed Data Directory: {PROCESSED_DATA_DIR}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
