import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch

class BookGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BookGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        return x
