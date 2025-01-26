
# # models/fastflow.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class ConditionalNormalization(nn.Module):
#     def __init__(self, in_channels, latent_dim):
#         super().__init__()
#         self.batch_norm = nn.BatchNorm2d(in_channels, affine=False, momentum=0.01)
#         self.scale_transform = nn.Sequential(
#             nn.Linear(latent_dim, in_channels),
#             nn.Dropout(0.3)
#         )
#         self.bias_transform = nn.Sequential(
#             nn.Linear(latent_dim, in_channels),
#             nn.Dropout(0.3)
#         )
        
#     def forward(self, x, latent):
#         normalized = self.batch_norm(x)
#         scale = self.scale_transform(latent).unsqueeze(-1).unsqueeze(-1)
#         bias = self.bias_transform(latent).unsqueeze(-1).unsqueeze(-1)
#         return normalized * scale + bias


# class CouplingLayer(nn.Module):
#     def __init__(self, in_channels, hidden_dim, latent_dim):
#         super().__init__()
#         # Simplify the conv_net architecture - single conv layer with reduced parameters
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(in_channels // 2, hidden_dim // 2, 1),  # Reduced hidden dimensions
#             nn.BatchNorm2d(hidden_dim // 2, momentum=0.01),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(hidden_dim // 2, in_channels, 1)  # Direct mapping to output
#         )
        
#         # Simplified conditional normalization
#         self.cond_norm = nn.Sequential(
#             nn.BatchNorm2d(hidden_dim // 2, momentum=0.01, affine=False),
#             nn.Dropout(0.2)  # Reduced dropout rate
#         )
        
#         # Simple conditioning layer
#         self.condition = nn.Linear(latent_dim, hidden_dim // 2)
        
#     def forward(self, x, latent, reverse=False):
#         x1, x2 = torch.chunk(x, 2, dim=1)
        
#         # Simplified forward pass
#         h = self.conv_net[0:2](x1)  # First conv and batch norm
        
#         # Simple conditional normalization
#         cond = self.condition(latent).unsqueeze(-1).unsqueeze(-1)
#         h = self.cond_norm(h) * cond
        
#         # Final convolution
#         h = self.conv_net[2:](h)
        
#         # Simple affine transformation
#         scale, shift = torch.chunk(h, 2, dim=1)
#         scale = torch.sigmoid(scale + 2.)
        
#         if not reverse:
#             y2 = x2 * scale + shift
#         else:
#             y2 = (x2 - shift) / scale
        
#         return torch.cat([x1, y2], dim=1)

# class FastFlow(nn.Module):
#     def __init__(self, in_channels, hidden_dim, latent_dim, n_flows=2):
#         super().__init__()
#         self.reshape_channels = 80  # Must be even for coupling layers
#         self.spatial_size = 4
        
#         # Feature normalization
#         self.feature_norm = nn.BatchNorm1d(in_channels, momentum=0.01, affine=False)
#         self.dropout = nn.Dropout(0.3)
        
#         # Feature conditioning
#         self.feature_conditioning = nn.Sequential(
#             nn.Linear(in_channels, in_channels),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(in_channels, in_channels),
#             nn.LayerNorm(in_channels)
#         )
        
#         # Verify dimensions
#         expected_size = self.reshape_channels * self.spatial_size * self.spatial_size
#         assert expected_size == in_channels, \
#             f"Reshape dimensions {self.reshape_channels}*{self.spatial_size}*{self.spatial_size} " \
#             f"({expected_size}) != input features {in_channels}"
        
#         print(f"FastFlow initialized with:")
#         print(f"- Input features: {in_channels}")
#         print(f"- Reshape dimensions: {self.reshape_channels} channels × {self.spatial_size}×{self.spatial_size} spatial")
#         print(f"- Hidden dimension: {hidden_dim}")
#         print(f"- Latent dimension: {latent_dim}")
        
#         # Initialize flow layers
#         self.flows = nn.ModuleList([
#             CouplingLayer(self.reshape_channels, hidden_dim, latent_dim)
#             for _ in range(n_flows)
#         ])
        
#         # Temperature scaling
#         self.temperature = nn.Parameter(torch.ones(1) * 0.1)
    
#     def forward(self, x, latent, reverse=False):
#         batch_size = x.size(0)
        
#         # Feature preprocessing
#         x = self.feature_norm(x)
#         x = self.dropout(x)
#         x = self.feature_conditioning(x)
        
#         # Reshape to [B, C, H, W]
#         x = x.view(batch_size, self.reshape_channels, self.spatial_size, self.spatial_size)
        
#         if not reverse:
#             for flow in self.flows:
#                 x = flow(x, latent)
#             x = x * torch.sigmoid(self.temperature)
#         else:
#             x = x / torch.sigmoid(self.temperature)
#             for flow in reversed(self.flows):
#                 x = flow(x, latent, reverse=True)
        
#         return x

# models/fastflow.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.conv(x)
        return x * attention_weights

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        
        attention = (avg_out + max_out).view(b, c, 1, 1)
        return x * attention

class AttentionCouplingLayer(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_dim // 2, 1),
            nn.BatchNorm2d(hidden_dim // 2, momentum=0.01),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim // 2, in_channels, 1)
        )
        
        # Add both spatial and channel attention
        self.spatial_attention = SpatialAttention(hidden_dim // 2)
        self.channel_attention = ChannelAttention(hidden_dim // 2)
        
        self.cond_norm = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2, momentum=0.01, affine=False),
            nn.Dropout(0.2)
        )
        
        self.condition = nn.Linear(latent_dim, hidden_dim // 2)
        
    def forward(self, x, latent, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Initial convolution
        h = self.conv_net[0:2](x1)
        
        # Apply attention mechanisms
        h = self.spatial_attention(h)
        h = self.channel_attention(h)
        
        # Conditional normalization
        cond = self.condition(latent).unsqueeze(-1).unsqueeze(-1)
        h = self.cond_norm(h) * cond
        
        # Final convolution
        h = self.conv_net[2:](h)
        
        scale, shift = torch.chunk(h, 2, dim=1)
        scale = torch.sigmoid(scale + 2.)
        
        if not reverse:
            y2 = x2 * scale + shift
        else:
            y2 = (x2 - shift) / scale
        
        return torch.cat([x1, y2], dim=1)

class AttentionFastFlow(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, n_flows=2):
        super().__init__()
        self.reshape_channels = 80
        self.spatial_size = 4
        
        self.feature_norm = nn.BatchNorm1d(in_channels, momentum=0.01, affine=False)
        self.dropout = nn.Dropout(0.3)
        
        # Enhanced feature conditioning with attention
        self.feature_conditioning = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels)
        )
        
        # Global attention for feature maps
        self.global_spatial_attention = SpatialAttention(self.reshape_channels)
        self.global_channel_attention = ChannelAttention(self.reshape_channels)
        
        # Initialize attention-enhanced flow layers
        self.flows = nn.ModuleList([
            AttentionCouplingLayer(self.reshape_channels, hidden_dim, latent_dim)
            for _ in range(n_flows)
        ])
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x, latent, reverse=False):
        batch_size = x.size(0)
        
        # Feature preprocessing
        x = self.feature_norm(x)
        x = self.dropout(x)
        x = self.feature_conditioning(x)
        
        # Reshape to [B, C, H, W]
        x = x.view(batch_size, self.reshape_channels, self.spatial_size, self.spatial_size)
        
        # Apply global attention
        x = self.global_spatial_attention(x)
        x = self.global_channel_attention(x)
        
        if not reverse:
            for flow in self.flows:
                x = flow(x, latent)
            x = x * torch.sigmoid(self.temperature)
        else:
            x = x / torch.sigmoid(self.temperature)
            for flow in reversed(self.flows):
                x = flow(x, latent, reverse=True)
        
        return x