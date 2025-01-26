# models/feature_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, version='efficientnet-b0'):
        super().__init__()
        # Load pretrained EfficientNet
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
                # Freeze all base layers
        for param in self.efficientnet.parameters():
            param.requires_grad = False
            
        # Only fine-tune the final few layers (last 2 blocks)
        for param in self.efficientnet.features[-2:].parameters():
            param.requires_grad = True
            
        # Remove the classifier
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # Get the actual feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)  # Assuming 256x256 input size
            features = self.efficientnet(dummy_input)
            self.feature_dims = features.view(1, -1).size(1)
            print(f"Actual feature dimensions: {self.feature_dims}")
        
    def forward(self, x):
        # Extract features and flatten
        features = self.efficientnet(x)
        return features.view(features.size(0), -1)