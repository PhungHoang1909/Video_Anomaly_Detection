import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.mem import MemVADModel, compute_anomaly_score
from data.preprocessing.prepare_ped2 import prepare_data
from config import Config

def normalize_scores(scores):
    """Min-max normalize scores to [0,1] range"""
    min_score = scores.min()
    max_score = scores.max()
    return (scores - min_score) / (max_score - min_score + 1e-8)

def sequence_scores_to_frame_scores(sequence_scores, sequence_length):
    """
    Convert sequence-level scores to frame-level scores by averaging overlapping sequences
    
    Args:
        sequence_scores: numpy array of scores for each sequence
        sequence_length: length of each sequence
        
    Returns:
        numpy array of frame-level scores
    """
    num_sequences = len(sequence_scores)
    num_frames = num_sequences + sequence_length - 1
    
    # Initialize arrays to store sum and count of scores for each frame
    frame_score_sums = np.zeros(num_frames)
    frame_score_counts = np.zeros(num_frames)
    
    # Add sequence scores to corresponding frames
    for i, score in enumerate(sequence_scores):
        frame_score_sums[i:i + sequence_length] += score
        frame_score_counts[i:i + sequence_length] += 1
    
    # Average the scores
    frame_scores = frame_score_sums / frame_score_counts
    
    return frame_scores

def evaluate_model(model, test_loader, gt_labels, device, results_dir, sequence_length):
    """
    Evaluate model performance and compute AUC score
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        gt_labels: Ground truth labels
        device: Computation device
        results_dir: Directory to save evaluation results
        sequence_length: Length of each sequence
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            reconstruction, memory_scores = model(data)
            
            # Compute anomaly scores
            batch_scores = compute_anomaly_score(data, reconstruction, memory_scores)
            all_scores.append(batch_scores.cpu().numpy())
    
    # Concatenate all scores
    all_scores = np.concatenate(all_scores, axis=0)
    
    # Convert sequence scores to frame scores
    frame_scores = sequence_scores_to_frame_scores(all_scores, sequence_length)
    
    # Ensure same length as ground truth
    if len(frame_scores) > len(gt_labels):
        frame_scores = frame_scores[:len(gt_labels)]
    elif len(frame_scores) < len(gt_labels):
        gt_labels = gt_labels[:len(frame_scores)]
    
    print(f"Frame scores shape: {frame_scores.shape}")
    print(f"Ground truth labels shape: {gt_labels.shape}")
    
    # Normalize scores to [0,1]
    normalized_scores = normalize_scores(frame_scores)
    
    # Compute ROC AUC
    roc_auc = roc_auc_score(gt_labels, normalized_scores)
    
    # Compute PR AUC
    precision, recall, _ = precision_recall_curve(gt_labels, normalized_scores)
    pr_auc = auc(recall, precision)
    
    # Plot ROC and PR curves
    plot_curves(gt_labels, normalized_scores, results_dir)
    
    # Save scores
    np.save(results_dir / 'anomaly_scores.npy', normalized_scores)
    
    return {
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'anomaly_scores': normalized_scores
    }

def plot_curves(gt_labels, scores, results_dir):
    """Plot and save ROC and PR curves"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(gt_labels, scores)
    roc_auc = roc_auc_score(gt_labels, scores)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    # Plot PR curve
    precision, recall, _ = precision_recall_curve(gt_labels, scores)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(results_dir / 'evaluation_curves.png')
    plt.close()

def main():
    # Load configuration
    config = Config()
    
    # Create directories
    config.make_dirs()
    
    # Load ground truth labels
    gt_labels_path = Path(config.DATA_PATH) / 'test_labels.npy'
    gt_labels = np.load(gt_labels_path)
    print(f"Loaded ground truth labels: {gt_labels.shape}")
    
    # Load model
    model = MemVADModel(
        input_size=config.INPUT_SIZE[0],
        patch_size=config.PATCH_SIZE,
        in_channels=config.IN_CHANNELS,
        num_frames=config.SEQUENCE_LENGTH,
        feature_dim=config.EMBED_DIM,
        num_memory=1024
    ).to(config.DEVICE)
    
    # Load trained model weights
    checkpoint_path = Path(config.CHECKPOINT_DIR) / 'final_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded trained model")
    
    # Prepare test data
    _, test_dataset = prepare_data(
        config.DATA_PATH,
        config.INPUT_SIZE,
        config.SEQUENCE_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Evaluate model
    print("Starting evaluation...")
    metrics = evaluate_model(
        model,
        test_loader,
        gt_labels,
        config.DEVICE,
        Path(config.CHECKPOINT_DIR),
        config.SEQUENCE_LENGTH
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"ROC AUC: {metrics['ROC_AUC']:.4f}")
    print(f"PR AUC: {metrics['PR_AUC']:.4f}")
    
    # Save metrics
    results_path = Path(config.CHECKPOINT_DIR) / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        for metric, value in metrics.items():
            if metric != 'anomaly_scores':
                f.write(f"{metric}: {value:.4f}\n")

if __name__ == '__main__':
    main()