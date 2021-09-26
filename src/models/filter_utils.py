import numpy as np
import torch

def get_edge_filter(input_dim):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
    edge_k = torch.from_numpy(kernel)
    edge_3dfilter = edge_k.repeat(input_dim,1,1,1,1)
    return edge_3dfilter

class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.squeeze()

class EdgeUtils:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.edge_3dfilter = get_edge_filter(input_dim)

    def forward(self, x):
        with torch.no_grad():
            edge_x = torch.nn.functional.conv3d(x, weight=self.edge_3dfilter, padding=1, groups=self.input_dim)
        return edge_x[:,:,1:-1,:,:]


