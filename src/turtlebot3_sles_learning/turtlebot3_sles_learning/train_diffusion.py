#!/usr/bin/env python3
"""
Train Diffusion Policy for imitation learning.
Maps (robot_v, robot_w, target_x_robot_frame, target_y_robot_frame, lidar_scan) -> (control_v, control_w)
Uses diffusion process to learn action distribution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import math

class ImitationLearningDataset(Dataset):
    """Dataset for imitation learning (same as MLP version)."""
    
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


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionDenoisingNetwork(nn.Module):
    """Denoising network for diffusion policy."""
    
    def __init__(self, action_dim=2, condition_dim=364, timestep_embed_dim=128, 
                 hidden_dims=[256, 128, 64], dropout=0.1):
        """
        Initialize denoising network.
        
        Args:
            action_dim: Dimension of action space (2 for control_v, control_w)
            condition_dim: Dimension of condition (364 for inputs)
            timestep_embed_dim: Dimension of timestep embedding
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(DiffusionDenoisingNetwork, self).__init__()
        
        # Timestep embedding
        self.timestep_embed = SinusoidalPositionEmbeddings(timestep_embed_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, timestep_embed_dim),
            nn.ReLU()
        )
        
        # Input: [noisy_action (2), timestep_embed (128), condition (364)]
        input_dim = action_dim + timestep_embed_dim + condition_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer: predict noise (same dimension as action)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, noisy_action, timestep, condition):
        """
        Forward pass.
        
        Args:
            noisy_action: (batch, action_dim) - noisy action at timestep t
            timestep: (batch,) - diffusion timestep
            condition: (batch, condition_dim) - input condition
        
        Returns:
            predicted_noise: (batch, action_dim) - predicted noise
        """
        # Embed timestep
        t_emb = self.timestep_embed(timestep)
        t_emb = self.timestep_mlp(t_emb)
        
        # Concatenate inputs
        x = torch.cat([noisy_action, t_emb, condition], dim=1)
        
        # Predict noise
        predicted_noise = self.network(x)
        
        return predicted_noise


class DiffusionPolicy:
    """Diffusion policy for action generation."""
    
    def __init__(self, model, num_timesteps=100, beta_start=0.0001, beta_end=0.02, 
                 device='cpu'):
        """
        Initialize diffusion policy.
        
        Args:
            model: Denoising network
            num_timesteps: Number of diffusion timesteps T
            beta_start: Starting noise level
            beta_end: Ending noise level
            device: Device to run on
        """
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Create noise schedule (linear)
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), 
                                            self.alpha_cumprod[:-1]])
        
        # Precompute values for sampling
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.posterior_variance = self.beta * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
    
    def add_noise(self, actions, timesteps):
        """
        Add noise to actions at given timesteps.
        
        Args:
            actions: (batch, action_dim) - clean actions
            timesteps: (batch,) - timesteps for each sample
        
        Returns:
            noisy_actions: (batch, action_dim) - noisy actions
            noise: (batch, action_dim) - noise that was added
        """
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timesteps].unsqueeze(1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].unsqueeze(1)
        
        noise = torch.randn_like(actions)
        noisy_actions = sqrt_alpha_cumprod_t * actions + sqrt_one_minus_alpha_cumprod_t * noise
        
        return noisy_actions, noise
    
    def sample(self, condition, num_samples=1):
        """
        Sample actions from diffusion model.
        
        Args:
            condition: (batch, condition_dim) - input condition
            num_samples: Number of samples to generate
        
        Returns:
            actions: (batch, action_dim) - sampled actions
        """
        self.model.eval()
        
        batch_size = condition.shape[0]
        action_dim = 2
        
        # Start from pure noise
        actions = torch.randn((batch_size, action_dim)).to(self.device)
        
        # Iterative denoising
        with torch.no_grad():
            for t in tqdm(range(self.num_timesteps - 1, -1, -1), 
                         desc="Sampling", leave=False):
                timesteps = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
                
                # Predict noise
                predicted_noise = self.model(actions, timesteps, condition)
                
                # Compute coefficients
                alpha_t = self.alpha[t]
                alpha_cumprod_t = self.alpha_cumprod[t]
                beta_t = self.beta[t]
                
                if t > 0:
                    # Denoise step
                    pred_action_0 = (actions - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                    actions = torch.sqrt(alpha_t) * pred_action_0 + torch.sqrt(1.0 - alpha_t) * predicted_noise
                else:
                    # Final step
                    actions = (actions - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        return actions


def fast_sample(diffusion_policy, condition, num_steps=20):
    """
    Fast sampling with fewer steps (DDIM-like) for evaluation.
    
    Args:
        diffusion_policy: Diffusion policy
        condition: Input condition (batch, condition_dim)
        num_steps: Number of sampling steps (default: 20 for fast inference)
    
    Returns:
        actions: Sampled actions (batch, action_dim)
    """
    diffusion_policy.model.eval()
    
    batch_size = condition.shape[0]
    action_dim = 2
    
    # Create step indices (evenly spaced from T-1 to 0)
    step_indices = np.linspace(diffusion_policy.num_timesteps - 1, 0, num_steps).astype(int)
    
    # Start from noise
    actions = torch.randn((batch_size, action_dim)).to(diffusion_policy.device)
    
    with torch.no_grad():
        for i, t in enumerate(step_indices):
            timesteps = torch.full((batch_size,), t, dtype=torch.long).to(diffusion_policy.device)
            
            # Predict noise
            predicted_noise = diffusion_policy.model(actions, timesteps, condition)
            
            # Compute coefficients
            alpha_cumprod_t = diffusion_policy.alpha_cumprod[t]
            
            if i < len(step_indices) - 1:
                # Denoise step (DDIM-like)
                t_next = step_indices[i + 1]
                alpha_cumprod_t_next = diffusion_policy.alpha_cumprod[t_next]
                
                pred_action_0 = (actions - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                actions = torch.sqrt(alpha_cumprod_t_next) * pred_action_0 + torch.sqrt(1.0 - alpha_cumprod_t_next) * predicted_noise
            else:
                # Final step
                actions = (actions - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
    
    return actions


def train_epoch(diffusion_policy, dataloader, optimizer, device):
    """Train for one epoch."""
    diffusion_policy.model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        batch_size = targets.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(0, diffusion_policy.num_timesteps, (batch_size,)).to(device)
        
        # Add noise to actions
        noisy_actions, noise = diffusion_policy.add_noise(targets, timesteps)
        
        # Predict noise
        optimizer.zero_grad()
        predicted_noise = diffusion_policy.model(noisy_actions, timesteps, inputs)
        
        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(diffusion_policy, dataloader, device, return_metrics=False):
    """Validate model."""
    diffusion_policy.model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            batch_size = targets.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(0, diffusion_policy.num_timesteps, (batch_size,)).to(device)
            
            # Add noise to actions
            noisy_actions, noise = diffusion_policy.add_noise(targets, timesteps)
            
            # Predict noise
            predicted_noise = diffusion_policy.model(noisy_actions, timesteps, inputs)
            
            # Compute loss
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
            total_loss += loss.item()
            num_batches += 1
            
            if return_metrics:
                # Sample actions for evaluation using fast sampling (20 steps)
                # This matches actual inference performance
                sampled_actions = fast_sample(diffusion_policy, inputs, num_steps=20)
                all_outputs.append(sampled_actions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    if return_metrics:
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        mae = np.mean(np.abs(all_outputs - all_targets), axis=0)
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
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'num_timesteps': 100,  # Diffusion timesteps (train with full schedule, use fewer at inference)
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'hidden_dims': [256, 128, 64],
        'timestep_embed_dim': 128,
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'models_diffusion'
    }
    
    print("=" * 60)
    print("Diffusion Policy Training Configuration")
    print("=" * 60)
    print(f"Architecture: Denoising Network")
    print(f"  Input: [noisy_action(2) + timestep_embed(128) + condition(364)]")
    print(f"  Hidden: {config['hidden_dims']}")
    print(f"  Output: noise(2)")
    print(f"Diffusion timesteps: {config['num_timesteps']}")
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
    
    # Initialize denoising network
    condition_dim = train_dataset.inputs.shape[1]
    denoising_network = DiffusionDenoisingNetwork(
        action_dim=2,
        condition_dim=condition_dim,
        timestep_embed_dim=config['timestep_embed_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout']
    ).to(config['device'])
    
    print(f"\nDenoising Network architecture:")
    print(denoising_network)
    total_params = sum(p.numel() for p in denoising_network.parameters())
    trainable_params = sum(p.numel() for p in denoising_network.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize diffusion policy
    diffusion_policy = DiffusionPolicy(
        model=denoising_network,
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=config['device']
    )
    
    # Optimizer
    optimizer = optim.Adam(denoising_network.parameters(), lr=config['learning_rate'])
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
        train_loss = train_epoch(diffusion_policy, train_loader, optimizer, config['device'])
        train_losses.append(train_loss)
        
        # Validate with detailed metrics
        val_metrics = validate(diffusion_policy, val_loader, config['device'], return_metrics=True)
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
                'model_state_dict': denoising_network.state_dict(),
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
    denoising_network.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(diffusion_policy, test_loader, config['device'], return_metrics=True)
    
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
    plt.title('Diffusion Policy Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    print(f"\nTraining curves saved to {config['save_dir']}/training_curves.png")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
