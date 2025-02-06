import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_scatter import scatter_add
import torch_sparse

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layer_num):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, x):
        logits = self.net(x)
        return logits

class EdgeRAU(nn.Module):
    def __init__(self, props):
        super(EdgeRAU, self).__init__()
        self.props = props

        self.num_path_mlp_hidden_layers = props.num_path_mlp_hidden_layers
        self.num_path_mlp_input_dim = 2
        self.num_path_mlp_hidden_dim = props.num_path_mlp_hidden_dim
        self.num_path_mlp_output_dim = 1
        self.path_mlp = MLP(
            input_dim=self.num_path_mlp_input_dim, output_dim=self.num_path_mlp_output_dim,
            hidden_dim=self.num_path_mlp_hidden_dim,
            hidden_layer_num=self.num_path_mlp_hidden_layers
        )
        self.edge_mlp = MLP(
                input_dim=3, output_dim=1, hidden_dim=props.num_edge_mlp_hidden_dim, hidden_layer_num=props.num_edge_mlp_hidden_layers
            )
        

    
    def forward(self, capacities, props, tms, tms_pred, pair_tms_hist, paths_to_edges, num_for_loops):
        """
            Args:
                capacities: torch.Tensor, shape: (batch_size, num_edges)
                tms: torch.Tensor, shape: (batch_size, num_paths)
                tms_pred: torch.Tensor, shape: (batch_size, num_paths)
                pair_tms_hist: torch.Tensor, shape: (batch_size, num_pairs, hist_len)
                paths_to_edges: torch.Tensor, shape: (num_paths, num_edges)
                edges_to_paths: torch.Tensor, shape: (num_edges, num_paths)

        """
        num_paths = paths_to_edges.shape[0]
        num_paths_per_pair = self.props.num_paths_per_pair
        batch_size = tms_pred.shape[0]
        paths_to_edges = paths_to_edges.coalesce()
        edge_indices = paths_to_edges.indices()[1]
        path_indices = paths_to_edges.indices()[0]
        num_edges = paths_to_edges.shape[1]
        self.props = props

        if self.props.dynamic:
            min_capacities, _ = torch_scatter.scatter_min(capacities[:, edge_indices], path_indices, dim=1, dim_size=num_paths) # shape: (batch_size, num_paths)
        else:
            min_capacities, _ = torch_scatter.scatter_min(capacities[:, edge_indices], path_indices, dim=1, dim_size=num_paths) # shape: (1, num_paths)
            capacities = capacities.repeat(batch_size, 1) # shape: (batch_size, num_edges)
            min_capacities = min_capacities.repeat(batch_size, 1) # shape: (batch_size, num_paths)

        path_input = torch.cat(
            (min_capacities.unsqueeze(-1), tms_pred.unsqueeze(-1)), dim=-1
        ) # shape: (batch_size, num_paths, 2)
        path_embeddings = self.path_mlp(path_input) # shape: (batch_size, num_paths, 1)
        path_embeddings = path_embeddings.squeeze(-1) # shape: (batch_size, num_paths)
        edge_lambda = torch.matmul(path_embeddings, paths_to_edges) # shape: (batch_size, num_edges)
        edge_lambda = F.sigmoid(edge_lambda) # shape: (batch_size, num_edges)
        if self.props.failures is not None and self.props.mode == "test":
            edge_lambda[:, self.props.failures] = 1e9
        for i in range(num_for_loops):
            path_sum_lambda = torch.matmul(paths_to_edges, edge_lambda.t()).t().reshape(batch_size, -1, num_paths_per_pair) # shape: (batch_size, num_pairs, num_paths_per_pair)
            split_ratios = torch.nn.functional.softmin(path_sum_lambda, dim=-1).reshape(batch_size, -1) # shape: (batch_size, num_paths)
            flow_on_paths = tms_pred * split_ratios # shape: (batch_size, num_paths)
            flow_on_edges = torch.matmul(flow_on_paths, paths_to_edges) # shape: (batch_size, num_edges)
            link_util = flow_on_edges / capacities # shape: (batch_size, num_edges)
            inf_mask = torch.isinf(link_util)
            nan_mask = torch.isnan(link_util)

            link_util[inf_mask] = 1e4
            link_util[nan_mask] = 0

            mlu, mlu_indices = torch.max(link_util, dim=-1)
            # edge_mlp_input = torch.cat((
            #     link_util.unsqueeze(-1),
            #     mlu.reshape(-1, 1, 1).repeat(1, num_edges, 1),
            #     edge_lambda.unsqueeze(-1),
            #     capacities.unsqueeze(-1),
            # ),dim=-1) # shape: (batch_size, num_edges, 4)
            edge_mlp_input = torch.cat((
                link_util.unsqueeze(-1) / mlu.reshape(-1, 1, 1).repeat(1, num_edges, 1),
                edge_lambda.unsqueeze(-1),
                capacities.unsqueeze(-1),
            ),dim=-1) # shape: (batch_size, num_edges, 3)

            delta_edge_lambda = self.edge_mlp(edge_mlp_input) # shape: (batch_size, num_edges, 1)
            delta_edge_lambda = torch.tanh(delta_edge_lambda)
            edge_lambda = edge_lambda + delta_edge_lambda.squeeze(-1)
          
        
        path_sum_lambda = torch.matmul(paths_to_edges, edge_lambda.t()).t().reshape(batch_size, -1, num_paths_per_pair) # shape: (batch_size, num_pairs, num_paths_per_pair)
        split_ratios = torch.nn.functional.softmin(path_sum_lambda, dim=-1).reshape(batch_size, -1)
        flow_on_paths = tms * split_ratios
        flow_on_edges = torch.matmul(flow_on_paths, paths_to_edges)
        link_util = flow_on_edges / capacities
        inf_mask = torch.isinf(link_util)
        nan_mask = torch.isnan(link_util)

        link_util[inf_mask] = 1e4
        link_util[nan_mask] = 0
        return link_util