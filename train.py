# # train.py
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import os
# import logging
# from tqdm import tqdm

# from vetro.models.feature_extractor import FeatureExtractor
# from vetro.models.fastflow import FastFlow
# from vetro.utils.dataset import FiberglassDataset
# from vetro.utils.visualization import save_loss_plot, visualize_predictions

# def train(config):
#     # Setup logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     # Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f'Using device: {device}')
    
#     # Create datasets and dataloaders
#     train_dataset = FiberglassDataset(config['data_root'], is_train=True,
#                                     preprocessing=config['preprocessing'])
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
#                             shuffle=True, num_workers=4)
    
#     # Initialize models
#     feature_extractor = FeatureExtractor(version=config['efficientnet_version'])
#     fastflow = FastFlow(in_channels=feature_extractor.feature_dims,
#                        hidden_dim=config['hidden_dim'],
#                        latent_dim=config['latent_dim'])
    
#     feature_extractor = feature_extractor.to(device)
#     fastflow = fastflow.to(device)
    
#     # Optimizer
#     optimizer = optim.Adam(list(feature_extractor.parameters()) + 
#                           list(fastflow.parameters()),
#                           lr=config['learning_rate'])
    
#     # Learning rate scheduler
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                     factor=0.5, patience=5,
#                                                     verbose=True)
    
#     # Training loop
#     best_loss = float('inf')
#     patience_counter = 0
#     train_losses = []
    
#     for epoch in range(config['epochs']):
#         feature_extractor.train()
#         fastflow.train()
        
#         total_loss = 0
#         pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
#         for batch in pbar:
#             images = batch.to(device)
            
#             # Extract features
#             features = feature_extractor(images)  # Shape: [B, feature_dims]
            
#             # Generate random latent vectors
#             latent = torch.randn(images.size(0), config['latent_dim']).to(device)
            
#             # Forward pass through FastFlow - features are automatically reshaped inside FastFlow
#             transformed = fastflow(features, latent)
            
#             # Compute MSE loss
#             target = torch.zeros_like(transformed)  # Assuming Gaussian prior centered at 0
#             loss = torch.mean((transformed - target) ** 2)
    
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             pbar.set_postfix({'loss': loss.item()})
        
#         avg_loss = total_loss / len(train_loader)
#         train_losses.append(avg_loss)
        
#         logger.info(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
#         # Learning rate scheduling
#         scheduler.step(avg_loss)
        
#         # Early stopping
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             patience_counter = 0
#             # Save best model
#             torch.save({
#                 'feature_extractor': feature_extractor.state_dict(),
#                 'fastflow': fastflow.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'loss': best_loss
#             }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
#         else:
#             patience_counter += 1
            
#         if patience_counter >= config['early_stopping_patience']:
#             logger.info('Early stopping triggered')
#             break
        
#         # Save checkpoint periodically
#         if (epoch + 1) % config['save_frequency'] == 0:
#             torch.save({
#                 'feature_extractor': feature_extractor.state_dict(),
#                 'fastflow': fastflow.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'loss': avg_loss
#             }, os.path.join(config['checkpoint_dir'],
#                           f'checkpoint_epoch_{epoch+1}.pth'))
    
#     # Save loss plot
#     save_loss_plot(train_losses, os.path.join(config['output_dir'],
#                                              'training_loss.png'))

###### MSE 

# # train.py
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import os
# import logging
# from tqdm import tqdm

# from vetro.models.feature_extractor import FeatureExtractor
# from vetro.models.fastflow import FastFlow
# from vetro.utils.dataset import FiberglassDataset
# from vetro.utils.visualization import save_loss_plot, visualize_predictions

# def train(config):
#     # Setup logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     # Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f'Using device: {device}')
    
#     # Create datasets and dataloaders
#     train_dataset = FiberglassDataset(config['data_root'], is_train=True,
#                                     preprocessing=config['preprocessing'])
#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
#                             shuffle=True, num_workers=4)
    
#     # Initialize models
#     feature_extractor = FeatureExtractor(version=config['efficientnet_version'])
#     fastflow = FastFlow(in_channels=feature_extractor.feature_dims,
#                        hidden_dim=config['hidden_dim'],
#                        latent_dim=config['latent_dim'])
    
#     feature_extractor = feature_extractor.to(device)
#     fastflow = fastflow.to(device)
    
#     # Initialize MSE Loss
#     #mse_loss = nn.MSELoss()
#     mse_loss = nn.MSELoss(reduction='none')
    
#     # Optimizer
#     optimizer = optim.Adam(list(feature_extractor.parameters()) + 
#                           list(fastflow.parameters()),
#                           lr=config['learning_rate'],
#                           weight_decay=1e-5) #L2 regulaizer
    
#     # Learning rate scheduler
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 
#         mode='min',
#         factor=0.5, 
#         patience=5,
#         verbose=True,
#         min_lr=1e-6
#     )
    
#     # Training loop
#     best_loss = float('inf')
#     patience_counter = 0
#     train_losses = []
    
#     logger.info("Starting training with zero-centered target...")
    
#     for epoch in range(config['epochs']):
#         feature_extractor.train()
#         fastflow.train()
        
#         total_loss = 0
#         pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
#         for batch in pbar:
#             images = batch.to(device)
            
#             # Extract features
#             features = feature_extractor(images)
            
#             # Generate random latent vectors
#             latent = torch.randn(images.size(0), config['latent_dim']).to(device)
            
#             # Forward pass through FastFlow
#             transformed = fastflow(features, latent)
            
#             # Use zero-centered target
#             target = torch.zeros_like(transformed).to(device)
            
#             # Compute MSE loss with stability measures
#             loss = mse_loss(transformed, target)
            
#             # Average over all dimensions
#             loss = torch.mean(loss)
            
#             # Add L1 regularization for sparsity
#             l1_lambda = 1e-5
#             l1_reg = sum(p.abs().sum() for p in fastflow.parameters())
#             loss = loss + l1_lambda * l1_reg
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(
#                 list(feature_extractor.parameters()) + 
#                 list(fastflow.parameters()), 
#                 max_norm=1.0
#             )
            
#             optimizer.step()
            
#             total_loss += loss.item()
#             pbar.set_postfix({'loss': loss.item()})
        
#         avg_loss = total_loss / len(train_loader)
#         train_losses.append(avg_loss)
        
#         logger.info(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
#         # Learning rate scheduling
#         scheduler.step(avg_loss)
        
#         # Early stopping with checkpointing
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             patience_counter = 0
#             # Save best model
#             torch.save({
#                 'feature_extractor': feature_extractor.state_dict(),
#                 'fastflow': fastflow.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'loss': best_loss
#             }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
#             logger.info(f'Saved new best model with loss: {best_loss:.4f}')
#         else:
#             patience_counter += 1
            
#         if patience_counter >= config['early_stopping_patience']:
#             logger.info('Early stopping triggered')
#             break
    
#     # Save loss plot
#     save_loss_plot(train_losses, os.path.join(config['output_dir'],
#                                              'training_loss.png'))

# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import logging
from tqdm import tqdm
import numpy as np
from vetro.utils.metrics import calculate_metrics

from vetro.models.feature_extractor import FeatureExtractor
#from vetro.models.fastflow import FastFlow
from vetro.models.fastflow import AttentionFastFlow as FastFlow
from vetro.utils.dataset import FiberglassDataset
from vetro.utils.visualization import save_loss_plot, save_validation_plot

def train(config):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create datasets and dataloaders
    full_train_dataset = FiberglassDataset(config['data_root'], is_train=True,
                                         preprocessing=config['preprocessing'])
    
    # Split training data into train and validation sets (80/20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    logger.info(f'Training set size: {train_size}, Validation set size: {val_size}')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=4)
    
    # Initialize models
    feature_extractor = FeatureExtractor(version=config['efficientnet_version'])
    fastflow = FastFlow(in_channels=feature_extractor.feature_dims,
                       hidden_dim=config['hidden_dim'],
                       latent_dim=config['latent_dim'])
    
    feature_extractor = feature_extractor.to(device)
    fastflow = fastflow.to(device)
    
    # Initialize MSE Loss with reduction='none' to get per-element losses
    mse_loss = nn.MSELoss(reduction='none')
    
    # Optimizer with gradient clipping
    optimizer = optim.Adam(list(feature_extractor.parameters()) + 
                          list(fastflow.parameters()),
                          lr=config['learning_rate'],
                          weight_decay=1e-5)  # Added L2 regularization
    
    # Learning rate scheduler based on validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5, 
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    def validation_step():
        feature_extractor.eval()
        fastflow.eval()
        total_val_loss = 0
        val_scores = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch.to(device)
                
                # Extract features
                features = feature_extractor(images)
                
                # Multiple forward passes for robust validation
                n_samples = 4  # Fewer samples than test time for speed
                batch_scores = []
                
                for _ in range(n_samples):
                    latent = torch.randn(images.size(0), config['latent_dim']).to(device)
                    transformed = fastflow(features, latent)
                    target = torch.zeros_like(transformed).to(device)
                    
                    # Calculate MSE score
                    mse_score = mse_loss(transformed, target)
                    score = torch.mean(mse_score, dim=[1, 2, 3])
                    batch_scores.append(score)
                
                # Average scores across samples
                avg_score = torch.stack(batch_scores).mean(0)
                val_scores.extend(avg_score.cpu().numpy())
                
                # Calculate validation loss
                loss = torch.mean(mse_score)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        return avg_val_loss, np.array(val_scores)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    logger.info("Starting training with validation...")
    
    for epoch in range(config['epochs']):
        # Training phase
        feature_extractor.train()
        fastflow.train()
        
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} - Training')
        
        for batch in pbar:
            images = batch.to(device)
            
            # Extract features
            features = feature_extractor(images)
            
            # Generate random latent vectors
            latent = torch.randn(images.size(0), config['latent_dim']).to(device)
            
            # Forward pass through FastFlow
            transformed = fastflow(features, latent)
            
            # Use zero-centered target
            target = torch.zeros_like(transformed).to(device)
            
            # Compute MSE loss with stability measures
            loss = mse_loss(transformed, target)
            loss = torch.mean(loss)
            
            # Add L1 regularization for sparsity
            l1_lambda = 1e-5
            l1_reg = sum(p.abs().sum() for p in fastflow.parameters())
            loss = loss + l1_lambda * l1_reg
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(feature_extractor.parameters()) + 
                list(fastflow.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'train_loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        avg_val_loss, val_scores = validation_step()
        val_losses.append(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}:')
        logger.info(f'  Training Loss: {avg_train_loss:.4f}')
        logger.info(f'  Validation Loss: {avg_val_loss:.4f}')
        
        # Learning rate scheduling based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early stopping with checkpointing based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'fastflow': fastflow.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            logger.info(f'  Saved new best model with validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            
        if patience_counter >= config['early_stopping_patience']:
            logger.info('Early stopping triggered')
            break
        
        # Periodic checkpointing
        if (epoch + 1) % config['save_frequency'] == 0:
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'fastflow': fastflow.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save training and validation loss plots
    save_validation_plot(
        train_losses, 
        val_losses, 
        os.path.join(config['output_dir'], 'loss_curves.png')
    )

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")