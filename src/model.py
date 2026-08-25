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


class SequenceAggregator(nn.Module):
    """Aggregates a sequence of slice features into a single study-level feature vector."""
    def __init__(self, input_dim, hidden_dim=256, num_layers=1):
        super(SequenceAggregator, self).__init__()
        
        # Bidirectional GRU to capture spatial relations between adjacent slices
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # The output feature dimension is doubled due to bidirectional accumulation (forward + backward)
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        # Input shape x: (Batch, SequenceLength, InputDim) -> e.g. (B, 32, 1280)
        
        # 1. Pass the sequence through the GRU
        # gru_out shape: (Batch, SequenceLength, HiddenDim * 2)
        gru_out, _ = self.gru(x)
        
        # 2. Pool the sequence features along the sequence/depth dimension (dim=1)
        # Average pooling over the 32 slices
        pooled_out = torch.mean(gru_out, dim=1)  # Shape: (Batch, HiddenDim * 2)
        
        return pooled_out


class RSNAKneeModel(nn.Module):
    """Unified single-view PyTorch model for RSNA Knee Abnormality Detection."""
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, num_classes=12):
        super(RSNAKneeModel, self).__init__()
        
        # 1. 2.5D Slice Encoder
        self.encoder = SliceEncoder(backbone_name=backbone_name, pretrained=pretrained)
        
        # 2. Sequence Aggregator (takes features from CNN output dim)
        self.aggregator = SequenceAggregator(input_dim=self.encoder.feature_dim)
        
        # 3. Final Multi-Label Classification Head
        # Outputting logits directly (we will apply Sigmoid during loss calculation/inference)
        self.classifier = nn.Linear(self.aggregator.output_dim, num_classes)

    def forward(self, x):
        # Input shape: (Batch, Channels, Depth, Height, Width) -> e.g. (B, 1, 32, 256, 256)
        
        # 1. Encode slices: (Batch, Depth, FeatureDim)
        features = self.encoder(x)
        
        # 2. Aggregate sequence: (Batch, AggregatorOutputDim)
        aggregated_features = self.aggregator(features)
        
        # 3. Classify: (Batch, 12)
        logits = self.classifier(aggregated_features)
        
        return logits
