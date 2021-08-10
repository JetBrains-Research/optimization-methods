import torch
import torch.nn as nn


def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):  # smth from google
    a = a + torch.eye(int(a.size(-1)), device=a.device)
    d_norm = calc_degree_matrix_norm(a)
    return torch.bmm(torch.bmm(d_norm, a), d_norm)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim, bias=use_bias), nn.ReLU())

    def forward(self, nodes, edges):
        l_norm = edges + torch.eye(edges.size(-1), device=edges.device)  # create_graph_lapl_norm(edges)
        out = torch.bmm(l_norm, nodes)  # sums each node vector with its nearest neighbours
        return self.net(out)

