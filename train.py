import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.mem import MemVADModel, compute_anomaly_score
from data.preprocessing.prepare_ped2 import prepare_data
from config import Config

class AnomalyLoss(nn.Module):
    """Combined loss function for anomaly detection"""
    def __init__(self, memory_weight=0.5, consistency_weight=0.3):
        super().__init__()
        self.memory_weight = memory_weight
        self.consistency_weight = consistency_weight
        
    def forward(self, x, recon, memory_scores):
        # Reconstruction loss
        recon_loss = F.mse_loss(x, recon)
        
        # Ensure memory_scores is a list or process accordingly
        if isinstance(memory_scores, list):
            mem_scores = torch.stack(memory_scores, dim=1)
        else:
            mem_scores = memory_scores  # Directly use memory_scores if it's already a tensor
        
        # Memory consistency loss
        mem_loss = -torch.log(mem_scores + 1e-8).mean()
        
        # Temporal consistency loss
        temp_loss = F.mse_loss(
            x[:, 1:] - x[:, :-1],
            recon[:, 1:] - recon[:, :-1]
        )
        
        return (recon_loss + 
                self.memory_weight * mem_loss + 
                self.consistency_weight * temp_loss)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        reconstruction, memory_scores = model(data)
        
        # Compute loss
        loss = criterion(data, reconstruction, memory_scores)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % Config.LOG_INTERVAL == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def main():
    # Initialize model
    model = MemVADModel(
        input_size=Config.INPUT_SIZE[0],
        in_channels=Config.IN_CHANNELS,
        num_frames=Config.SEQUENCE_LENGTH,
        feature_dim=Config.EMBED_DIM,
    ).to(Config.DEVICE)
    
    # Prepare data
    train_dataset, test_dataset = prepare_data(
        Config.DATA_PATH,
        Config.INPUT_SIZE,
        Config.SEQUENCE_LENGTH
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    
    # Initialize loss and optimizer
    criterion = AnomalyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=Config.LEARNING_RATE * 0.01
    )
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch {epoch + 1}/{Config.EPOCHS}')
        
        train_loss = train_epoch(
            model, train_loader, criterion,
            optimizer, Config.DEVICE
        )
        
        print(f'Epoch {epoch + 1}, Average Loss: {train_loss:.4f}')
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pth')
            
        torch.save({
            'epoch': Config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, Config.CHECKPOINT_DIR / 'final_model.pth')

if __name__ == '__main__':
    main()