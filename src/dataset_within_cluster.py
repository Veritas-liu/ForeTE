import os
import numpy as np
from torch.utils.data import Dataset
from src.config import DATA_DIR
from src.snapshot_utils import Read_Snapshot
from src.cluster_utils import Cluster_Info

class DM_Dataset_within_Cluster(Dataset):
    def __init__(self, props, cluster, start, end):
        self.props = props
        self.cluster = cluster
        self.tm_list = []
        self.tm_hist_list = []
        self.capacity_list = []
        self.opt_list = []

        if self.props.failure_num == 0:
            opt_file = os.path.join(DATA_DIR, props.topo_name, 'Opt', f'{props.num_paths_per_pair}sp', f'{cluster}', 'opt_values.txt')
            if os.path.exists(opt_file):
                file = open(opt_file, 'r')
                opts = np.loadtxt(file, dtype=np.float32).ravel()
                file.close()
                opts = opts[start:end]
                if len(opts) < end-start:
                    opts = np.array([np.float32(1) for _ in range(start, end)])
            else:
                opts = np.array([np.float32(1) for _ in range(start, end)])
        else:
            opt_file = os.path.join(DATA_DIR, props.topo_name, 'Opt', f'{props.num_paths_per_pair}sp', f'{cluster}', f'opt_values_failures_{self.props.failure_num}.txt')
            opts = np.loadtxt(opt_file, dtype=np.float32).ravel()
            link_failures_path = os.path.join(DATA_DIR, props.topo_name,'link_failure',f"{self.props.failure_num}.npy")
            failure_links = np.load(link_failures_path)

        catalog_file = os.path.join(DATA_DIR, props.topo_name, 'Catalog', f'{cluster}', 'catalog_file.txt')
        catalog = np.loadtxt(catalog_file, dtype="U", delimiter=",").reshape(-1, 3)
        catalog = catalog[start:end]
        props.catalog_len = len(catalog)

        tm_history = []
        for idx,(snapshot_filename, opt_value) in enumerate(zip(catalog, opts)):
            topology_filename, pairs_filename, tm_filename = snapshot_filename
            if self.props.failure_num != 0:
                props.failures = failure_links[idx].tolist()
            snapshot = Read_Snapshot(props, topology_filename, pairs_filename, tm_filename)

            if len(tm_history) == props.hist_len:
                self.tm_list.append(snapshot.tm)
                self.tm_hist_list.append(np.stack(tm_history, axis=0))
                self.capacity_list.append(snapshot.capacities)
                self.opt_list.append(opt_value)

            tm_history.append(snapshot.tm)
            if len(tm_history) > props.hist_len:
                tm_history.pop(0)

        cluster_info = Cluster_Info(props, snapshot, self.cluster)
        self.paths_to_edges = cluster_info.paths_to_edges
    
    def __len__(self):
        return len(self.tm_list)
    
    def __getitem__(self, idx):
        return self.tm_list[idx], self.tm_hist_list[idx], self.capacity_list[idx], self.opt_list[idx]