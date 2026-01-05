import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.checkpoint import checkpoint

class PoseEmbeddingMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Flatten pose matrices
        x = x.view(x.size(0), -1)
        return self.mlp(x)

class VisioPoseModel(nn.Module):
    def __init__(self, embed_dim=128, use_checkpointing=False):
        super().__init__()
        self.vision = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        in_features = self.vision.heads.head.in_features
        self.vision.heads.head = nn.Linear(in_features, embed_dim)
        
        self.pose_encoder = PoseEmbeddingMLP(output_dim=embed_dim)
        self.use_checkpointing = use_checkpointing
        
    def _checkpoint_vision(self, x):
        def custom_forward(module):
            def _forward(*inputs):
                return module(*inputs)
            return _forward
            
        if self.use_checkpointing:
             x = self.vision._process_input(x)
             n = x.shape[0]
             
             batch_class_token = self.vision.class_token.expand(n, -1, -1)
             x = torch.cat([batch_class_token, x], dim=1)
             x = x + self.vision.encoder.pos_embedding
             
             for block in self.vision.encoder.layers:
                 x = checkpoint(custom_forward(block), x, use_reentrant=False)
                 
             x = self.vision.encoder.ln(x)
             x = x[:, 0]
             return self.vision.heads(x)
        else:
            return self.vision(x)
        
    def forward(self, image, pose):
        v_emb = self._checkpoint_vision(image)
        p_emb = self.pose_encoder(pose)
        
        # L2 norm for contrastive space
        v_emb = nn.functional.normalize(v_emb, p=2, dim=1)
        p_emb = nn.functional.normalize(p_emb, p=2, dim=1)
        
        return v_emb, p_emb
