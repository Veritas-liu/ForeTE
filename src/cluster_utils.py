import torch
import numpy as np
import os
from scipy.sparse import csr_matrix
from itertools import islice
import pickle
import networkx as nx

from src.snapshot_utils import Read_Snapshot
from src.config import DATA_DIR


class Cluster_Info:
    def __init__(self, props, snapshot: Read_Snapshot, cluster):
        self.props = props
        self.snapshot = snapshot
        self.cluster = cluster
        self.edges_map = {(i, j): eid for eid, (i, j) in enumerate(self.snapshot.graph.edges())}
        self.pij = self.compute_ksp_paths(self.props.num_paths_per_pair, self.snapshot.pairs)
        self.paths_to_edges = self.get_paths_to_edges_matrix(self.pij)
    
    def node_ids_to_edge_tuple(self, node_ids):
        return [(node1, node2) for node1, node2 in zip(node_ids, node_ids[1:])]
    
    def find_ksp_per_pair(self, graph, src, dst, k, weight=None):
        return list(islice(nx.shortest_simple_paths(graph, src, dst, weight=weight), k))
        
    def compute_ksp_paths(self, k, pairs):
        filepath = os.path.join(DATA_DIR, self.props.topo_name, 'Paths', f'{self.props.num_paths_per_pair}_paths_cluster_{self.cluster}.pkl')
        try:
            file = open(filepath, "rb")
            pij = pickle.load(file)
            self.num_pairs = len(pij)
            file.close()
        except:
            pij = dict()
            print(f"[Computing {k} Shortest Paths]")
            for src, dst in pairs:
                    all_paths = self.find_ksp_per_pair(self.snapshot.graph, src, dst, k)
                    # Force all pairs of nodes to have the same number of paths (K)
                    # If a pair of nodes has less than K paths, replicate the first paths
                    # until they are equal to K
                    while len(all_paths) != k:
                        all_paths.append(all_paths[0])
                    pij[(src, dst)] = [self.node_ids_to_edge_tuple(all_paths[i]) for i in range(k)]
            self.num_pairs = len(pairs)
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            file = open(filepath, "wb")
            pickle.dump(pij, file)
            file.close()
        return pij
    
    def get_paths_to_edges_matrix(self, pij: dict) -> torch.sparse_coo_tensor:
        filepath = os.path.join(DATA_DIR, self.props.topo_name, 'P2E', f'{self.props.num_paths_per_pair}_paths_cluster_{self.cluster}.pkl')
        try:
            paths_to_edges = torch.load(filepath, weights_only=False)
        except FileNotFoundError:
            row_indices = []
            col_indices = []
            data = []

            path_idx = 0
            for key in pij.keys():
                i, j = key
                for p in pij[(i, j)]:
                    p_ = [self.edges_map[e] for e in p]
                    for edge_idx in p_:
                        row_indices.append(path_idx) 
                        col_indices.append(edge_idx)
                        data.append(1)
                    path_idx += 1

            num_paths = path_idx
            paths_to_edges = csr_matrix((data, (row_indices, col_indices)), 
                                        shape=(num_paths, len(self.edges_map)))
            paths_to_edges = paths_to_edges.tocoo()
            paths_to_edges = torch.sparse_coo_tensor(np.vstack((paths_to_edges.row, paths_to_edges.col)), 
                            torch.FloatTensor(paths_to_edges.data), torch.Size(paths_to_edges.shape)).coalesce()
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            torch.save(paths_to_edges, filepath)
            
        return paths_to_edges