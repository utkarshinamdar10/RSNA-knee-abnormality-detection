import torch
import torch.nn as nn
import timm



class SliceEncoder(nn.Module):
    """Encodes a 3D MRI volume slice-by-slice using a pretrained 2D CNN backbone."""
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True):
        super(SliceEncoder, self).__init__()
        
        # Load the pretrained 2D backbone from timm
        # num_classes=0 tells timm to return the raw feature vector (pooling output)
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        
        # Determine the size of the feature vector output by the backbone
        self.feature_dim = self.backbone.num_features

    def forward(self, x):
        # Input shape x: (Batch, Channels, Depth, Height, Width) -> e.g. (B, 1, 32, 256, 256)
        batch_size, channels, depth, height, width = x.shape
        
        # 1. Permute and reshape to stack all slices together: (Batch * Depth, Channels, Height, Width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, Depth, Channels, H, W)
        x = x.view(batch_size * depth, channels, height, width)  # (B*D, C, H, W)
        
        # 2. Replicate single grayscale channel to 3 channels for pretrained RGB models
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)  # (B*D, 3, H, W)
            
        # 3. Extract feature vector per slice using the 2D CNN
        features = self.backbone(x)  # Shape: (B*D, FeatureDim)
        
        # 4. Reshape back into sequence shape: (Batch, Depth, FeatureDim)
        features = features.view(batch_size, depth, self.feature_dim)
        
        return features


