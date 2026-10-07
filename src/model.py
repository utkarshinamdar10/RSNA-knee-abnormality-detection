import torch
import torch.nn as nn
import timm

LABEL_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]


class RSNAKneeModel(nn.Module):
    """
    EfficientNet-B0 backbone that processes each MRI slice independently,
    then mean-pools slice features to produce 12 abnormality logits.

    Input : (B, 1, D, H, W)  — 3D volume from RSNAKneeDataset
    Output: (B, 12)           — raw logits for 12 abnormalities
    """
    def __init__(self, backbone='efficientnet_b0', num_classes=12, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0, global_pool='avg'
        )
        feat_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.squeeze(1).reshape(B * D, 1, H, W)    # (B*D, 1, H, W)
        feats = self.encoder(x)                       # (B*D, feat_dim)
        feats = feats.reshape(B, D, -1).mean(dim=1)   # (B, feat_dim)
        return self.head(feats)                        # (B, 12)
