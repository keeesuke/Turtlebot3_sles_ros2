#!/usr/bin/env python3
"""
Train MLP for imitation learning.
Maps (robot_v, robot_w, target_pos_in_robot_frame, lidar_scan) -> (control_v, control_w)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class ImitationLearningDataset(Dataset):
    """Dataset for imitation learning."""
    
    def __init__(self, npz_file):
        """
        Load dataset from npz file.
        
        Args:
            npz_file: Path to npz file containing:
                - lidar_scans: (N, 360)
                - states: (N, 2) - [v, w]
                - target_positions: (N, 2) - target in robot frame
                - control_linear: (N,)
                - control_angular: (N,)
        """
        data = np.load(npz_file, allow_pickle=True)
        
        # Concatenate inputs: [robot_v, robot_w, target_x, target_y, lidar_scan]
        self.inputs = np.concatenate([
            data['states'],  # (N, 2)
            data['target_positions'],  # (N, 2)
            data['lidar_scans']  # (N, 360)
        ], axis=1).astype(np.float32)
        
        # Outputs: [control_linear, control_angular]
        self.outputs = np.stack([
            data['control_linear'],
            data['control_angular']
        ], axis=1).astype(np.float32)
        
        print(f"Dataset loaded: {len(self.inputs)} samples")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Output shape: {self.outputs.shape}")
        print(f"  Input range: [{self.inputs.min():.4f}, {self.inputs.max():.4f}]")
        print(f"  Output range: [{self.outputs.min():.4f}, {self.outputs.max():.4f}]")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.inputs[idx]), torch.FloatTensor(self.outputs[idx])
    

class MLP(nn.Module):
    """Multi-Layer Perceptron for imitation learning."""
    
    def __init__(self, input_dim=364, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.1):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension (robot_state + target + lidar = 2 + 2 + 360 = 364)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (control_v, control_w = 2)
            dropout: Dropout probability
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate(model, dataloader, criterion, device, return_metrics=False):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            if return_metrics:
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    if return_metrics:
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        mae = np.mean(np.abs(all_outputs - all_targets), axis=0)  # Per output dimension
        mse_per_dim = np.mean((all_outputs - all_targets) ** 2, axis=0)
        rmse_per_dim = np.sqrt(mse_per_dim)
        
        # R² score
        ss_res = np.sum((all_targets - all_outputs) ** 2, axis=0)
        ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2, axis=0)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse_per_dim,
            'r2': r2
        }
        return metrics
    else:
        return avg_loss

def main():
    # Configuration
    config = {
        'train_file': 'train_dataset.npz',
        'val_file': 'val_dataset.npz',
        'test_file': 'test_dataset.npz',
        'batch_size': 256,
        'learning_rate': 1e-3,
        'num_epochs': 20,
        'hidden_dims': [256, 128, 64],  # 3 hidden layers
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'models'
    }
    
    print("=" * 60)
    print("MLP Training Configuration")
    print("=" * 60)
    print(f"Architecture: Input(364) -> {config['hidden_dims']} -> Output(2)")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Device: {config['device']}")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ImitationLearningDataset(config['train_file'])
    val_dataset = ImitationLearningDataset(config['val_file'])
    test_dataset = ImitationLearningDataset(config['test_file'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Initialize model
    input_dim = train_dataset.inputs.shape[1]
    model = MLP(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        output_dim=2,
        dropout=config['dropout']
    ).to(config['device'])
    
    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        train_losses.append(train_loss)
        
        # Validate with detailed metrics
        val_metrics = validate(model, val_loader, criterion, config['device'], return_metrics=True)
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print detailed evaluation results
        print(f"\nTrain Loss (MSE): {train_loss:.6f}")
        print(f"Val Loss (MSE): {val_loss:.6f}")
        print(f"\nValidation Metrics:")
        print(f"  Control Linear (v):")
        print(f"    MAE:  {val_metrics['mae'][0]:.6f}")
        print(f"    RMSE: {val_metrics['rmse'][0]:.6f}")
        print(f"    R²:   {val_metrics['r2'][0]:.6f}")
        print(f"  Control Angular (w):")
        print(f"    MAE:  {val_metrics['mae'][1]:.6f}")
        print(f"    RMSE: {val_metrics['rmse'][1]:.6f}")
        print(f"    R²:   {val_metrics['r2'][1]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"\n  -> New best model saved! (Val Loss: {val_loss:.6f})")
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, config['device'], return_metrics=True)
    
    print(f"\nTest Loss (MSE): {test_metrics['loss']:.6f}")
    print(f"\nTest Metrics:")
    print(f"  Control Linear (v):")
    print(f"    MAE:  {test_metrics['mae'][0]:.6f}")
    print(f"    RMSE: {test_metrics['rmse'][0]:.6f}")
    print(f"    R²:   {test_metrics['r2'][0]:.6f}")
    print(f"  Control Angular (w):")
    print(f"    MAE:  {test_metrics['mae'][1]:.6f}")
    print(f"    RMSE: {test_metrics['rmse'][1]:.6f}")
    print(f"    R²:   {test_metrics['r2'][1]:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    print(f"\nTraining curves saved to {config['save_dir']}/training_curves.png")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
