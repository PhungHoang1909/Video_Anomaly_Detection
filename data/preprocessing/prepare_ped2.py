import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import scipy.io

class PED2Preprocessor:
    def __init__(self, data_path, output_size=(256, 256), sequence_length=16):
        self.data_path = Path(data_path)
        self.output_size = output_size
        self.sequence_length = sequence_length
        
    def load_frames(self, frames_dir):
        """Load frames from a directory."""
        frames = []
        frame_files = sorted(frames_dir.glob('*.jpg'), 
                           key=lambda x: int(x.stem.split('_')[-1]))  # Handle different frame naming patterns
        
        if not frame_files:
            raise ValueError(f"No frames found in directory: {frames_dir}")
            
        print(f"Found {len(frame_files)} frames in {frames_dir}")
        
        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.output_size)
            frames.append(frame)
            
        if not frames:
            raise ValueError(f"No valid frames could be loaded from {frames_dir}")
            
        return np.array(frames)
    
    def create_sequences(self, frames):
        """Create overlapping sequences of frames."""
        if len(frames) < self.sequence_length:
            raise ValueError(f"Not enough frames ({len(frames)}) to create a sequence of length {self.sequence_length}")
            
        sequences = []
        for i in range(0, len(frames) - self.sequence_length + 1):
            sequence = frames[i:i + self.sequence_length]
            sequences.append(sequence)
        return np.array(sequences)
    
    def process_dataset(self, split='training'):
        """Process entire dataset for given split."""
        split_path = self.data_path / split / "frames"
        
        if not split_path.exists():
            raise ValueError(f"Split path does not exist: {split_path}")
            
        print(f"Processing {split} data from {split_path}")
        folders = list(sorted(split_path.glob('*')))
        
        if not folders:
            raise ValueError(f"No data folders found in {split_path}")
            
        processed_data = []
        total_folders = len(folders)
        
        for idx, folder in enumerate(folders, 1):
            if folder.is_dir():
                try:
                    print(f"Processing folder {idx}/{total_folders}: {folder.name}")
                    frames = self.load_frames(folder)
                    sequences = self.create_sequences(frames)
                    processed_data.append(sequences)
                    print(f"Successfully processed {len(sequences)} sequences from {folder.name}")
                except Exception as e:
                    print(f"Error processing folder {folder}: {str(e)}")
                    continue
        
        if not processed_data:
            raise ValueError(f"No data could be processed from {split_path}")
            
        return np.concatenate(processed_data)
    
class PED2Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.FloatTensor(data).permute(0, 1, 4, 2, 3) / 255.0
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sequence = self.data[idx]
        if self.transform:
            sequence = self.transform(sequence)
        return sequence
    
def verify_dataset_structure(data_path):
    """Verify the dataset directory structure."""
    data_path = Path(data_path)
    
    # Check if base path exists
    if not data_path.exists():
        raise ValueError(f"Dataset path does not exist: {data_path}")
        
    # Check training directory
    train_path = data_path / "training" / "frames"
    if not train_path.exists():
        raise ValueError(f"Training frames directory not found at: {train_path}")
        
    # Check testing directory
    test_path = data_path / "testing" / "frames"
    if not test_path.exists():
        raise ValueError(f"Testing frames directory not found at: {test_path}")
        
    # Check for frame folders
    train_folders = list(train_path.glob('*'))
    test_folders = list(test_path.glob('*'))
    
    if not train_folders:
        raise ValueError(f"No training folders found in {train_path}")
    if not test_folders:
        raise ValueError(f"No testing folders found in {test_path}")
        
    print(f"Dataset structure verification complete:")
    print(f"- Found {len(train_folders)} training folders")
    print(f"- Found {len(test_folders)} testing folders")
    
    return True
    
def prepare_data(data_path, output_size=(256, 256), sequence_length=16):
    """Main function to prepare the dataset."""
    try:
        # Verify dataset structure
        verify_dataset_structure(data_path)
        
        preprocessor = PED2Preprocessor(data_path, output_size, sequence_length)
        
        # Process training data
        print("\nProcessing training data...")
        train_data = preprocessor.process_dataset('training')
        print(f"Training data shape: {train_data.shape}")
        
        # Process testing data
        print("\nProcessing testing data...")
        test_data = preprocessor.process_dataset('testing')
        print(f"Testing data shape: {test_data.shape}")
        
        # Convert to proper format
        train_dataset = PED2Dataset(train_data)
        test_dataset = PED2Dataset(test_data)
        
        return train_dataset, test_dataset
        
    except Exception as e:
        print(f"\nError preparing dataset: {str(e)}")
        print("\nPlease verify that your dataset follows this structure:")
        print("""
        ped2/
        ├── training/
        │   └── frames/
        │       ├── 01/
        │       │   ├── 001.jpg
        │       │   ├── 002.jpg
        │       │   └── ...
        │       ├── 02/
        │       └── ...
        └── testing/
            └── frames/
                ├── 01/
                │   ├── 001.jpg
                │   ├── 002.jpg
                │   └── ...
                ├── 02/
                └── ...
        """)
        raise
    
if __name__ == '__main__':
    data_path = '/kaggle/input/mem-vad/MEM_Video-Anomaly-Detection/datasets/ped2'
    try:
        train_dataset, test_dataset = prepare_data(data_path)
        print("Dataset preparation successful!")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Testing dataset size: {len(test_dataset)}")
    except Exception as e:
        print(f"Failed to prepare dataset: {str(e)}")