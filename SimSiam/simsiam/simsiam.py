import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet18, resnet50


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
    
class SimSiam(nn.Module):
    def __init__(self, backbone=None):
        super(SimSiam, self).__init__()
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

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}
    