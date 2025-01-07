import scipy.io
import numpy as np
from pathlib import Path
from typing import Tuple, List
from config import Config

def convert_range_to_labels(start: int, end: int, total_frames: int) -> np.ndarray:
    """
    Convert anomaly range to frame-level binary labels.
    
    Args:
        start: Start frame of anomaly (1-based index)
        end: End frame of anomaly (1-based index)
        total_frames: Total number of frames in sequence
        
    Returns:
        Binary array where 1 indicates anomaly frame
    """
    labels = np.zeros(total_frames)
    # Convert to 0-based indexing and ensure within bounds
    start_idx = max(0, start - 1)  # Matlab uses 1-based indexing
    end_idx = min(total_frames, end)
    labels[start_idx:end_idx] = 1
    return labels

def process_ped2_dataset(dataset_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Process the UCSD Ped2 dataset and extract ground truth labels from ped2.mat
    
    Args:
        dataset_path: Path to the ped2 dataset root directory
    
    Returns:
        Tuple containing:
        - numpy array of ground truth labels (1 for anomaly, 0 for normal)
        - list of frame paths
    """
    # Load the .mat file
    mat_path = dataset_path / 'ped2.mat'
    mat_data = scipy.io.loadmat(str(mat_path))
    
    print("Keys in mat file:", mat_data.keys())
    print("Shape of gt data:", mat_data['gt'].shape)
    
    # Get testing frames paths
    test_dir = dataset_path / 'testing' / 'frames'
    frame_paths = []
    all_labels = []
    
    # Process each testing sequence
    for seq_dir in sorted(test_dir.glob('[0-9][0-9]')):
        # Get frame paths for this sequence
        seq_frames = sorted(seq_dir.glob('*.jpg'))
        n_frames = len(seq_frames)
        frame_paths.extend([str(f) for f in seq_frames])
        
        # Get sequence number (1-based index)
        seq_num = int(seq_dir.name)
        
        try:
            # Get ground truth ranges for this sequence
            seq_gt = mat_data['gt'][0, seq_num-1]  # Shape: (2, 1)
            start_frame = seq_gt[0][0]  # Get start frame
            end_frame = seq_gt[1][0]    # Get end frame
            
            print(f"Sequence {seq_num}:")
            print(f"  Anomaly range: {start_frame} to {end_frame}")
            print(f"  Number of frames: {n_frames}")
            
            # Convert range to frame-level labels
            frame_labels = convert_range_to_labels(start_frame, end_frame, n_frames)
            
            print(f"  Created labels shape: {frame_labels.shape}")
            print(f"  Number of anomalous frames: {np.sum(frame_labels)}")
            
            all_labels.extend(frame_labels)
            
        except Exception as e:
            print(f"Error processing sequence {seq_num}:")
            print(f"  seq_gt type: {type(seq_gt)}")
            print(f"  seq_gt: {seq_gt}")
            print(f"  Error: {str(e)}")
            raise
    
    final_labels = np.array(all_labels)
    print(f"Final labels shape: {final_labels.shape}")
    return final_labels, frame_paths

def save_ground_truth(dataset_path: Path) -> Path:
    """
    Process the Ped2 dataset and save ground truth labels as .npy file
    
    Args:
        dataset_path: Path to the ped2 dataset root directory
        
    Returns:
        Path to the saved ground truth labels file
    """
    # Process dataset and get labels
    labels, _ = process_ped2_dataset(dataset_path)
    
    # Create output path
    output_path = dataset_path / 'test_labels.npy'
    
    # Save labels
    np.save(output_path, labels)
    print(f"Saved ground truth labels to {output_path}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of anomalous frames: {np.sum(labels)}")
    
    return output_path

def verify_and_prepare_dataset(config: Config) -> Path:
    """
    Verify dataset structure and prepare ground truth labels
    
    Args:
        config: Configuration object with dataset paths
        
    Returns:
        Path to ground truth labels file
    """
    dataset_path = config.DATA_PATH
    
    # Verify basic structure
    testing_path = dataset_path / 'testing' / 'frames'
    if not testing_path.exists():
        raise FileNotFoundError(f"Testing frames directory not found at {testing_path}")
        
    if not (dataset_path / 'ped2.mat').exists():
        raise FileNotFoundError(f"ped2.mat not found in {dataset_path}")
    
    # Check if ground truth already exists
    gt_path = dataset_path / 'test_labels.npy'
    if not gt_path.exists():
        gt_path = save_ground_truth(dataset_path)
        
    return gt_path