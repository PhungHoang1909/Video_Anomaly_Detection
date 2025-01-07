# run_processing.py
from pathlib import Path
from process_ped2_labels import verify_and_prepare_dataset
from config import Config

if __name__ == "__main__":
    config = Config()
    try:
        # This will create test_labels.npy in your ped2 dataset directory
        gt_path = verify_and_prepare_dataset(config)
        print(f"Successfully created ground truth labels at: {gt_path}")
    except Exception as e:
        print(f"Error processing dataset: {e}")