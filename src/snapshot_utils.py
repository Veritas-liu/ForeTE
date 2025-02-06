import os
import json
import networkx as nx
import torch
import pickle
import numpy as np

from src.config import DATA_DIR


class Read_Snapshot:
    def __init__(self, props, topology_filename, pairs_filename, tm_filename):
        
        self.props = props
        self.graph, self.capacities = self.read_graph_from_json(topology_filename)
        self.pairs = self.read_pairs_from_pkl(pairs_filename)
        self.tm = self.read_tm(tm_filename)
        if props.pred:
            self.tm_pred = self.read_tm_pred(tm_filename)
        else:
            self.tm_pred = self.tm


    def read_graph_from_json(self, filename):
        json_file_path = os.path.join(DATA_DIR, self.props.topo_name, 'Topology', filename)
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
        graph = nx.readwrite.json_graph.node_link_graph(json_data)

        capacities = [float(data['capacity']) for u, v, data in graph.edges(data=True)]
        capacities = torch.tensor(capacities, dtype=self.props.dtype)

        if self.props.failures is not None:
            for failure in self.props.failures:
                capacities[failure] = 0

        return graph, capacities
                    
    def read_pairs_from_pkl(self, filename):
        pkl_file_path = os.path.join(DATA_DIR, self.props.topo_name, 'Pairs', filename)
        with open(pkl_file_path, 'rb') as file:
            pairs = pickle.load(file)
        return pairs
    
    def read_tm(self, filename):
        tm_file_path = os.path.join(DATA_DIR, self.props.topo_name, 'TMs', filename)
        with open(tm_file_path, 'rb') as file:
            tm = pickle.load(file)
        tm = tm.astype(np.float32)
        tm = np.repeat(tm, self.props.num_paths_per_pair, axis=0)
        assert tm.shape[0] == len(self.pairs) * self.props.num_paths_per_pair
        return tm
    
    def read_tm_pred(self, filename):
        tm_pred_file_path = os.path.join(DATA_DIR, self.props.topo_name, 'TMs_pred', self.props.pred_method, filename)
        with open(tm_pred_file_path, 'rb') as file:
            tm_pred = pickle.load(file)
        tm_pred = tm_pred.astype(np.float32)
        tm_pred = np.repeat(tm_pred, self.props.num_paths_per_pair, axis=0)
        assert tm_pred.shape[0] == len(self.pairs) * self.props.num_paths_per_pair
        return tm_pred