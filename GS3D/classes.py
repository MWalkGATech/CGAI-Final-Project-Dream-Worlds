from torch import nn
import torch

from GS3D.utils import Image, Point3D

class Gaussians(nn.Module):
  def __init__(self, points: list[Point3D], device='cpu'):
    super().__init__()

    self.num_samples = len(points)

    colors = torch.zeros((self.num_samples, 3), device=device)
    centers = torch.zeros((self.num_samples, 3), device=device)

    for i, point in enumerate(points):
      colors[i] = point.rgb
      centers[i] = point.xyz

    self.centers = nn.Parameter(centers, requires_grad=True)
    self.alphas = nn.Parameter(torch.logit(torch.rand(self.num_samples, 1, device=device)), requires_grad=True)
    self.colors = nn.Parameter(colors, requires_grad=True)
    self.quaternions = nn.Parameter()
    self.sigmas = nn.Parameter()
    