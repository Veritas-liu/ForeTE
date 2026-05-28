import argparse
import os
import csv

def add_default_args(parser):

    parser.add_argument('--topo_name', type=str, default = 'Facebook_tor_a', help="Name of the topology, default is Facebook_tor_a")
    parser.add_argument('--device_id', type=int, default = 0, help="Device ID, default is 0")
    parser.add_argument('--num_paths_per_pair', type=int, default = 3, help="Number of paths per pair, default is 3")
    parser.add_argument('--pred', type=int, default = 0, help="Indicates whether to use predicted TMs, default is 0")
    parser.add_argument('--pred_method', type=str, default = 'MovAvg', help="Method for prediction, default is MovAvg")
    parser.add_argument('--hist_len', type=int, default = 12, help="Length of history for prediction, default is 12")
    parser.add_argument('--forecast_type', type=str, default='exp', choices=['exp', 'dnn', 'hist_max'], help="Forecast model type: exp, dnn, or hist_max")
    parser.add_argument('--forecast_alpha', type=float, default=0.5, help="Smoothing alpha for exponential forecast")
    parser.add_argument('--forecast_hidden_dim', type=int, default=512, help="Hidden dimension for DNN forecast model")
    parser.add_argument('--forecast_hidden_layers', type=int, default=2, help="Number of hidden layers for DNN forecast model")
    parser.add_argument('--desensitization_weight', type=float, default=0.0, help="Weight for desensitization term combining log(path_capacity) with lambda")
    parser.add_argument('--model_name', type=str, default = '', help="The custom name added to the end of the model name. Default is emplt")

    parser.add_argument('--mode', type=str, default = 'train', help="Mode of operation: train or test, default is train")
    parser.add_argument('--batch_size', type=int, default = 1, help="Batch size for training, default is 1")
    parser.add_argument('--epochs', type=int, default = 1, help="Number of training epochs, default is 1")
    parser.add_argument('--lr', type=float, default = 0.001, help="Learning rate, default is 0.001")
    parser.add_argument('--dtype', type=str, default='float32', help="Data type for the model, default is float32")
    parser.add_argument('--split_num', type=int, default = 1, help="Number of splits for the pairs, default is 1")
    
    parser.add_argument('--num_path_mlp_hidden_layers', type=int, default = 3, help="Number of hidden layers for path MLP, default is 3")
    parser.add_argument('--num_path_mlp_hidden_dim', type=int, default = 1, help="Hidden dimension for path MLP, default is 1")
    parser.add_argument('--num_edge_mlp_hidden_layers', type=int, default = 1, help="Number of hidden layers for edge MLP, default is 1")
    parser.add_argument('--num_edge_mlp_hidden_dim', type=int, default = 8, help="Hidden dimension for edge MLP, default is 8")
    parser.add_argument('--num_for_loops', type=int, default = 3, help="Number of loops for the RAU, default is 3")

    parser.add_argument('--dynamic', type=int, default=0, help="Indicates whether the topology is dynamic, default is 0")
    parser.add_argument('--train_clusters', type=int, nargs='+', help="List of clusters to train on")
    parser.add_argument('--train_start_indices', type=int, nargs='+', help="List of start indices for training")
    parser.add_argument('--train_end_indices', type=int, nargs='+', help="List of end indices for training")
    parser.add_argument('--val_clusters', type=int, nargs='+', help="List of clusters to val on")
    parser.add_argument('--val_start_indices', type=int, nargs='+', help="List of start indices for val")
    parser.add_argument('--val_end_indices', type=int, nargs='+', help="List of end indices for val")
    parser.add_argument('--test_cluster', type=int, help="Cluster to test on")
    parser.add_argument('--test_start_idx', type=int, help="Start index for testing")
    parser.add_argument('--test_end_idx', type=int, help="End index for testing")

    parser.add_argument('--failures', type=int, nargs='+', help="List of failures to test on")
    parser.add_argument('--failure_num',type=int, default=0)
    return parser

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    
    return parser.parse_args(args)

def save_results(args, file_path):
    file_exists = os.path.isfile(file_path)
    data = {
        "topo_name": args["topo_name"],
        "num_paths_per_pair": args["num_paths_per_pair"],
        "throughput (samples/s)": args['throughput'],
        "Memory Usage (GB)": args['memory_usage'],
        "Train clusters": args["train_clusters"],
    }
    with open(file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)