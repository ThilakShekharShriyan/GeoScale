import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PoseEmbeddingMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.mlp(x)

class VisioPoseModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.vision = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.vision.fc.in_features
        self.vision.fc = nn.Linear(in_features, embed_dim)

        self.pose_encoder = PoseEmbeddingMLP(output_dim=embed_dim)

    def forward(self, image, pose):
        v_emb = self.vision(image)
        p_emb = self.pose_encoder(pose)

        v_emb = nn.functional.normalize(v_emb, p=2, dim=1)
        p_emb = nn.functional.normalize(p_emb, p=2, dim=1)

        return v_emb, p_emb
