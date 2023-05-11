import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet18, resnet50
import numpy as np


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception
    
class ProjectionMLP(nn.Module):
    """Projection MLP f"""
    def __init__(self, in_features, h1_features, h2_features, out_features):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, h1_features),
            nn.BatchNorm1d(h1_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(h1_features, h2_features),
            nn.BatchNorm1d(h2_features),
            nn.ReLU(inplace=True)

        )
        self.l3 = nn.Sequential(
            nn.Linear(h1_features, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x 


class PredictionMLP(nn.Module):
    """Prediction MLP h"""
    def __init__(self, in_features, hidden_features, out_features):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
    
class FastSiam(nn.Module):
    def __init__(self, backbone=None):
        super(FastSiam, self).__init__()
        if backbone is None:
            backbone = resnet50()
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

        self.backbone = backbone
        
        self.projector = ProjectionMLP(backbone.output_dim, 2048, 2048, 2048)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.predictor = PredictionMLP(2048, 512, 2048)

    def forward(self, x1, views):
        f, h = self.encoder, self.predictor
        # x2 is the averaged encoding of the views
        embeddings_z_views = []
        embeddings_p_views = []
        for view in views:
            view = view.to('cuda', non_blocking=True)
            z = f(view)
            p = h(z)
            embeddings_z_views.append(z)
            embeddings_p_views.append(p)

        embeddings_z_views = torch.stack(embeddings_z_views, dim=0)
        embeddings_p_views = torch.stack(embeddings_p_views, dim=0)
        # average embeddings
        z2 = torch.mean(embeddings_z_views, dim=0) # TODO Check
        p2 = torch.mean(embeddings_p_views, dim=0)
        
        z1 = f(x1)
        p1 = h(z1)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}