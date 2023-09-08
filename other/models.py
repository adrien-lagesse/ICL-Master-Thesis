import torch
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_features: int, out_features: int, embedding_size: int) -> None:

        super(GCN, self).__init__()

        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        self.out = torch.nn.Linear(embedding_size*2, out_features)

    def forward(self, x: torch.FloatTensor, edge_index: torch.FloatTensor, batch_index:int) -> torch.FloatTensor:

        hidden = self.initial_conv(x, edge_index)
        hidden = torch.nn.functional.relu(hidden)

        hidden = self.conv1(hidden, edge_index)
        hidden = torch.nn.functional.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.nn.functional.relu(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = torch.nn.functional.relu(hidden)
          
        hidden = torch.cat([global_max_pool(hidden, batch_index), 
                            global_mean_pool(hidden, batch_index)], dim=1)
        out = self.out(hidden)

        return out