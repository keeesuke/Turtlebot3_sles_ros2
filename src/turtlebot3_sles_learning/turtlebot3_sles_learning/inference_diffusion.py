#!/usr/bin/env python3
"""
Inference script for diffusion policy.
Load trained model and generate control actions from inputs.
"""

import numpy as np
import torch
import torch.nn as nn
import math
from train_diffusion import DiffusionDenoisingNetwork, DiffusionPolicy, SinusoidalPositionEmbeddings

def load_diffusion_policy(model_path, device='cpu'):
    """
    Load trained diffusion policy.
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on
    
    Returns:
        diffusion_policy: Loaded diffusion policy
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize denoising network
    denoising_network = DiffusionDenoisingNetwork(
        action_dim=2,
        condition_dim=364,  # robot_v, robot_w, target_x, target_y, lidar(360)
        timestep_embed_dim=config.get('timestep_embed_dim', 128),
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    # Load weights
    denoising_network.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize diffusion policy
    diffusion_policy = DiffusionPolicy(
        model=denoising_network,
        num_timesteps=config.get('num_timesteps', 100),  # Load trained timesteps, but use fewer at inference
        beta_start=config.get('beta_start', 0.0001),
        beta_end=config.get('beta_end', 0.02),
        device=device
    )
    
    return diffusion_policy, config

def predict_action(diffusion_policy, robot_v, robot_w, target_x, target_y, lidar_scan, 
                   num_samples=1, fast_sampling=False, num_steps=None):
    """
    Predict control action from inputs.
    
    Args:
        diffusion_policy: Trained diffusion policy
        robot_v: Robot linear velocity
        robot_w: Robot angular velocity
        target_x: Target x position in robot frame
        target_y: Target y position in robot frame
        lidar_scan: Lidar scan (360 values)
        num_samples: Number of action samples to generate
        fast_sampling: If True, use fewer steps for faster inference (default: 20 steps)
        num_steps: Number of sampling steps (if fast_sampling=True, defaults to 20)
    
    Returns:
        action: (control_v, control_w) or array of actions if num_samples > 1
    """
    # Prepare condition
    condition = np.concatenate([
        [robot_v, robot_w],
        [target_x, target_y],
        lidar_scan
    ]).astype(np.float32)
    
    condition = torch.FloatTensor(condition).unsqueeze(0).to(diffusion_policy.device)
    
    # Repeat condition for multiple samples
    if num_samples > 1:
        condition = condition.repeat(num_samples, 1)
    
    # Sample actions
    if fast_sampling:
        # Use fewer steps for faster inference (default: 20 steps)
        if num_steps is None:
            num_steps = 20  # Default to 20 steps for fast inference
        actions = fast_sample(diffusion_policy, condition, num_steps)
    else:
        # Use full timesteps (100 steps, slower but higher quality)
        actions = diffusion_policy.sample(condition, num_samples=num_samples)
    
    actions = actions.cpu().numpy()
    
    if num_samples == 1:
        return actions[0]  # Return single action
    else:
        return actions  # Return array of actions

def fast_sample(diffusion_policy, condition, num_steps=10):
    """
    Fast sampling with fewer steps (DDIM-like).
    
    Args:
        diffusion_policy: Diffusion policy
        condition: Input condition
        num_steps: Number of sampling steps
    
    Returns:
        actions: Sampled actions
    """
    diffusion_policy.model.eval()
    
    batch_size = condition.shape[0]
    action_dim = 2
    
    # Create step indices
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
                # Denoise step
                t_next = step_indices[i + 1]
                alpha_cumprod_t_next = diffusion_policy.alpha_cumprod[t_next]
                
                pred_action_0 = (actions - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                actions = torch.sqrt(alpha_cumprod_t_next) * pred_action_0 + torch.sqrt(1.0 - alpha_cumprod_t_next) * predicted_noise
            else:
                # Final step
                actions = (actions - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
    
    return actions

def main():
    """Example usage."""
    import sys
    
    # Load model
    model_path = 'models_diffusion/best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading diffusion policy from {model_path}...")
    diffusion_policy, config = load_diffusion_policy(model_path, device)
    print("Model loaded successfully!")
    
    # Example inputs
    robot_v = 0.1
    robot_w = 0.05
    target_x = 1.0
    target_y = 0.5
    lidar_scan = np.ones(360) * 0.5  # Example lidar scan
    
    # Predict action (uses fast sampling with 20 steps by default)
    print("\nPredicting action...")
    action = predict_action(
        diffusion_policy,
        robot_v, robot_w, target_x, target_y, lidar_scan,
        num_samples=1,
        fast_sampling=True,  # Use fast sampling (20 steps instead of 100)
        num_steps=20
    )
    
    print(f"Predicted action: control_v={action[0]:.4f}, control_w={action[1]:.4f}")
    
    # Sample multiple actions (shows multi-modal distribution)
    print("\nSampling multiple actions (showing diversity)...")
    actions = predict_action(
        diffusion_policy,
        robot_v, robot_w, target_x, target_y, lidar_scan,
        num_samples=5
    )
    
    print("Sampled actions:")
    for i, a in enumerate(actions):
        print(f"  Sample {i+1}: control_v={a[0]:.4f}, control_w={a[1]:.4f}")

if __name__ == "__main__":
    main()
