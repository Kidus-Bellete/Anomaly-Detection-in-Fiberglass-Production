
# # test.py
# import torch
# import logging
# from torch.utils.data import DataLoader
# import numpy as np
# from tqdm import tqdm
# import os

# from vetro.models.feature_extractor import FeatureExtractor
# from vetro.models.fastflow import FastFlow
# from vetro.utils.dataset import FiberglassDataset
# from vetro.utils.metrics import calculate_metrics
# from vetro.utils.visualization import visualize_predictions, plot_roc_curve

# def test(config):
#     """
#     Test the anomaly detection model.
#     Args:
#         config (dict): Configuration dictionary containing model and data parameters
#     Returns:
#         dict: Dictionary containing evaluation metrics
#     """
#     # Setup logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     # Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f'Using device: {device}')
    
#     # Load test dataset
#     test_dataset = FiberglassDataset(config['data_root'], is_train=False,
#                                    preprocessing=config['preprocessing'])
#     test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
#                            shuffle=False, num_workers=4)
    
#     # Initialize models
#     feature_extractor = FeatureExtractor(version=config['efficientnet_version'])
#     fastflow = FastFlow(in_channels=feature_extractor.feature_dims,
#                        hidden_dim=config['hidden_dim'],
#                        latent_dim=config['latent_dim'])
    
#     # Load best model
#     model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(
#             f"No trained model found at {model_path}. "
#             "Please train the model first using --mode train"
#         )
    
#     checkpoint = torch.load(model_path, map_location=device)
#     feature_extractor.load_state_dict(checkpoint['feature_extractor'])
#     fastflow.load_state_dict(checkpoint['fastflow'])
    
#     feature_extractor = feature_extractor.to(device)
#     fastflow = fastflow.to(device)
    
#     # Evaluation mode
#     feature_extractor.eval()
#     fastflow.eval()
    
#     all_scores = []
#     all_labels = []
#     all_images = []
    
#     logger.info('Starting evaluation...')
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc='Testing'):
#             images = images.to(device)
            
#             # Extract features
#             features = feature_extractor(images)
            
#             # Multiple forward passes for robust prediction
#             n_samples = 8
#             scores = []
            
#             for _ in range(n_samples):
#                 latent = torch.randn(images.size(0),
#                                    config['latent_dim']).to(device)
#                 transformed = fastflow(features, latent)
#                 score = -torch.sum(transformed, dim=[1, 2, 3]).cpu().numpy()
#                 scores.append(score)
            
#             # Average scores across samples
#             avg_score = np.mean(scores, axis=0)
#             all_scores.extend(avg_score)
#             all_labels.extend(labels.numpy())
#             all_images.extend(images.cpu().numpy())
    
#     all_scores = np.array(all_scores)
#     all_labels = np.array(all_labels)
#     all_images = np.array(all_images)
    
#     # Calculate metrics
#     metrics = calculate_metrics(all_labels, all_scores)
    
#     # Log results
#     logger.info("\nTest Results:")
#     logger.info(f"AUC-ROC Score: {metrics['auroc']:.4f}")
#     logger.info(f"Average Precision: {metrics['avg_precision']:.4f}")
#     logger.info(f"F1 Score: {metrics['f1']:.4f}")
#     logger.info(f"Optimal Threshold: {metrics['threshold']:.4f}")
    
#     # Create output directory if it doesn't exist
#     os.makedirs(config['output_dir'], exist_ok=True)
    
#     # Visualize sample predictions
#     logger.info('Generating visualization of predictions...')
#     visualize_predictions(all_images, all_labels, all_scores,
#                         save_path=os.path.join(config['output_dir'],
#                                              'sample_predictions.png'))
    
#     # Plot ROC curve
#     logger.info('Generating ROC curve...')
#     plot_roc_curve(all_labels, all_scores,
#                   save_path=os.path.join(config['output_dir'],
#                                        'roc_curve.png'))
    
#     # Save predictions to CSV
#     try:
#         import pandas as pd
#         predictions_df = pd.DataFrame({
#             'true_label': all_labels,
#             'anomaly_score': all_scores,
#             'is_anomaly': all_scores > metrics['threshold']
#         })
#         predictions_df.to_csv(os.path.join(config['output_dir'],
#                                          'predictions.csv'),
#                             index=False)
#         logger.info(f"Predictions saved to {os.path.join(config['output_dir'], 'predictions.csv')}")
#     except Exception as e:
#         logger.warning(f"Could not save predictions to CSV: {str(e)}")
    
#     return metrics

# if __name__ == "__main__":
#     # This allows the script to be run directly for debugging
#     from vetro.main import main
#     main()

########## MSE error loss

# # test.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import logging
# from torch.utils.data import DataLoader
# import numpy as np
# from tqdm import tqdm
# import os

# from vetro.models.feature_extractor import FeatureExtractor
# from vetro.models.fastflow import FastFlow
# from vetro.utils.dataset import FiberglassDataset
# from vetro.utils.metrics import calculate_metrics
# from vetro.utils.visualization import visualize_predictions, plot_roc_curve

# def test(config):
#     """
#     Test the anomaly detection model using MSE-based scoring.
#     Args:
#         config (dict): Configuration dictionary containing model and data parameters
#     Returns:
#         dict: Dictionary containing evaluation metrics
    
#     """
#     # Setup logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     # Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f'Using device: {device}')
    
#     # Initialize MSE Loss
#     mse_loss = nn.MSELoss(reduction='none')
    
#     # Load test dataset
#     test_dataset = FiberglassDataset(config['data_root'], is_train=False,
#                                    preprocessing=config['preprocessing'])
#     test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
#                            shuffle=False, num_workers=4)
    
#     # Initialize and load models
#     feature_extractor = FeatureExtractor(version=config['efficientnet_version'])
#     fastflow = FastFlow(in_channels=feature_extractor.feature_dims,
#                        hidden_dim=config['hidden_dim'],
#                        latent_dim=config['latent_dim'])
    
#     model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
#     checkpoint = torch.load(model_path, map_location=device)
#     feature_extractor.load_state_dict(checkpoint['feature_extractor'])
#     fastflow.load_state_dict(checkpoint['fastflow'])
    
#     feature_extractor = feature_extractor.to(device)
#     fastflow = fastflow.to(device)
    
#     # Evaluation mode
#     feature_extractor.eval()
#     fastflow.eval()
    
#     all_scores = []
#     all_labels = []
#     all_images = []
    
#     logger.info('Starting evaluation with zero-centered MSE scoring...')
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc='Testing'):
#             images = images.to(device)
            
#             # Extract features
#             features = feature_extractor(images)
            
#             # Multiple forward passes for robust prediction
#             n_samples = 8
#             scores = []
            
#             for _ in range(n_samples):
#                 latent = torch.randn(images.size(0),
#                                    config['latent_dim']).to(device)
#                 transformed = fastflow(features, latent)
                
#                 # Use zero-centered target
#                 target = torch.zeros_like(transformed).to(device)
                
#                 # Calculate MSE score
#                 mse_score = mse_loss(transformed, target)
#                 score = torch.mean(mse_score, dim=[1, 2, 3]).cpu().numpy()
#                 scores.append(score)
            
#             # Average scores across samples
#             avg_score = np.mean(scores, axis=0)
#             all_scores.extend(avg_score)
#             all_labels.extend(labels.numpy())
#             all_images.extend(images.cpu().numpy())
    
#     all_scores = np.array(all_scores)
#     all_labels = np.array(all_labels)
    
#     # Calculate metrics
#     metrics = calculate_metrics(all_labels, all_scores)
    
#     logger.info("\nTest Results:")
#     logger.info(f"AUC-ROC Score: {metrics['auroc']:.4f}")
#     logger.info(f"Average Precision: {metrics['avg_precision']:.4f}")
#     logger.info(f"F1 Score: {metrics['f1']:.4f}")
#     logger.info(f"Optimal Threshold: {metrics['threshold']:.4f}")
    
#     return metrics

# if __name__ == "__main__":
#     # This allows the script to be run directly for debugging
#     from vetro.main import main
#     main()

# test.py
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from torchvision import transforms

from vetro.models.feature_extractor import FeatureExtractor
#from vetro.models.fastflow import FastFlow
from vetro.models.fastflow import AttentionFastFlow as FastFlow
from vetro.utils.dataset import FiberglassDataset
from vetro.utils.metrics import calculate_metrics
from vetro.utils.visualization import visualize_predictions, plot_roc_curve, plot_confusion_matrix, plot_score_distributions

def test_time_augmentation(image, n_aug=4):
    """Apply test-time augmentation to images."""
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ]
    
    aug_images = [image]
    for _ in range(n_aug - 1):
        aug_image = image.clone()
        for aug in augmentations:
            if torch.rand(1).item() > 0.5:
                aug_image = aug(aug_image)
        aug_images.append(aug_image)
    
    return torch.stack(aug_images)

def test(config):
    """
    Test the anomaly detection model using MSE-based scoring with test-time augmentation
    and ensemble predictions.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Initialize losses
    mse_loss = nn.MSELoss(reduction='none')
    
    # Load test dataset
    test_dataset = FiberglassDataset(config['data_root'], is_train=False,
                                   preprocessing=config['preprocessing'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=4)
    
    # Initialize models
    feature_extractor = FeatureExtractor(version=config['efficientnet_version'])
    fastflow = FastFlow(in_channels=feature_extractor.feature_dims,
                       hidden_dim=config['hidden_dim'],
                       latent_dim=config['latent_dim'])
    
    # Load best model
    model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    fastflow.load_state_dict(checkpoint['fastflow'])
    
    feature_extractor = feature_extractor.to(device)
    fastflow = fastflow.to(device)
    
    # Evaluation mode
    feature_extractor.eval()
    fastflow.eval()
    
    all_scores = []
    all_labels = []
    all_images = []
    
    # Parameters for robust evaluation
    n_tta = 4  # Number of test-time augmentations
    n_flows = 8  # Number of flow samples
    
    logger.info('Starting evaluation with test-time augmentation...')
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            batch_scores = []
            
            # Apply test-time augmentation
            for i in range(images.size(0)):
                image_scores = []
                aug_images = test_time_augmentation(images[i], n_tta).to(device)
                
                # Get features for all augmented versions
                aug_features = feature_extractor(aug_images)
                
                # Multiple forward passes for each augmented image
                for _ in range(n_flows):
                    latent = torch.randn(aug_images.size(0),
                                       config['latent_dim']).to(device)
                    transformed = fastflow(aug_features, latent)
                    target = torch.zeros_like(transformed).to(device)
                    
                    # Calculate MSE score
                    mse_score = mse_loss(transformed, target)
                    score = torch.mean(mse_score, dim=[1, 2, 3])
                    image_scores.append(score)
                
                # Average scores across flows and augmentations
                avg_score = torch.stack(image_scores).mean(dim=0).mean(dim=0)
                batch_scores.append(avg_score.cpu().numpy())
            
            all_scores.extend(batch_scores)
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Calculate metrics with uncertainty estimation
    metrics = calculate_metrics(all_labels, all_scores)
    
    # Detailed logging
    logger.info("\nTest Results:")
    logger.info(f"AUC-ROC Score: {metrics['auroc']:.4f}")
    logger.info(f"Average Precision: {metrics['avg_precision']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"Optimal Threshold: {metrics['threshold']:.4f}")
    
    # Save visualizations
    # Save all visualizations
    visualize_predictions(
        all_images, 
        all_labels, 
        all_scores,
        os.path.join(config['output_dir'], 'test_predictions.png')
    )

    plot_roc_curve(
        all_labels,
        all_scores,
        os.path.join(config['output_dir'], 'roc_curve.png')
    )

    plot_score_distributions(
        all_scores,
        all_labels,
        metrics['threshold'],
        os.path.join(config['output_dir'], 'score_distributions.png')
    )

    plot_confusion_matrix(
        all_labels,
        all_scores,
        metrics['threshold'],
        os.path.join(config['output_dir'], 'confusion_matrix.png')
    )

    return metrics

if __name__ == "__main__":
    from vetro.main import main
    main()