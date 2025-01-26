# # utils/visualization.py
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from sklearn.metrics import roc_curve

# def save_loss_plot(losses, save_path):
#     """Plot and save training loss curve"""
#     plt.figure(figsize=(10, 6))
#     plt.plot(losses, label='Training Loss')
#     plt.title('Training Loss Over Time')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.close()

# def visualize_predictions(images, labels, scores, save_path, n_samples=10):
#     """Visualize sample predictions with their anomaly scores"""
#     # Select random samples from each class
#     normal_idx = np.where(labels == 0)[0]
#     anomaly_idx = np.where(labels == 1)[0]
    
#     n_each = min(n_samples // 2, len(normal_idx), len(anomaly_idx))
#     selected_normal = np.random.choice(normal_idx, n_each, replace=False)
#     selected_anomaly = np.random.choice(anomaly_idx, n_each, replace=False)
    
#     selected_idx = np.concatenate([selected_normal, selected_anomaly])
    
#     # Create visualization
#     fig, axes = plt.subplots(2, n_each, figsize=(15, 6))
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
#     for i, idx in enumerate(selected_idx):
#         row = i // n_each
#         col = i % n_each
        
#         # Display image
#         axes[row, col].imshow(images[idx].transpose(1, 2, 0))
#         axes[row, col].axis('off')
        
#         # Add title with score
#         title = f'{"Anomaly" if labels[idx] == 1 else "Normal"}\nScore: {scores[idx]:.2f}'
#         axes[row, col].set_title(title)
    
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close()

# def plot_roc_curve(labels, scores, save_path):
#     """Plot and save ROC curve"""
#     fpr, tpr, _ = roc_curve(labels, scores)
#     auc_score = roc_auc_score(labels, scores)
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.close()

# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import logging

def normalize_image_for_display(image):
    """Normalize image to [0,1] range for display"""
    image = image - image.min()
    image = image / (image.max() + 1e-8)
    return image

def save_loss_plot(losses, save_path):
    """Plot and save training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(images, labels, scores, save_path, n_samples=10):
    """Visualize sample predictions with their anomaly scores"""
    # Select random samples from each class
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    
    n_each = min(n_samples // 2, len(normal_idx), len(anomaly_idx))
    selected_normal = np.random.choice(normal_idx, n_each, replace=False)
    selected_anomaly = np.random.choice(anomaly_idx, n_each, replace=False)
    
    selected_idx = np.concatenate([selected_normal, selected_anomaly])
    
    # Create visualization
    fig, axes = plt.subplots(2, n_each, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    for i, idx in enumerate(selected_idx):
        row = i // n_each
        col = i % n_each
        
        # Normalize image for display
        image = images[idx].transpose(1, 2, 0)
        image = normalize_image_for_display(image)
        
        # Display image
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
        
        # Add title with score
        title = f'{"Anomaly" if labels[idx] == 1 else "Normal"}\nScore: {scores[idx]:.2f}'
        axes[row, col].set_title(title)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curve(labels, scores, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_validation_plot(train_losses, val_losses, save_path):
    """Save a plot comparing training and validation losses."""
        
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_score_distributions(scores, labels, threshold, save_path):
    """Plot distribution of anomaly scores with decision threshold"""
    plt.figure(figsize=(10, 6))
    plt.hist(scores[labels == 0], bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(scores[labels == 1], bins=50, alpha=0.5, label='Anomalous', density=True)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(labels, scores, threshold, save_path):
    """Plot confusion matrix for anomaly detection"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    predictions = (scores > threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()