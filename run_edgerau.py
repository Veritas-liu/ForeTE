import sys
from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import os
import re
from time import time
from tqdm import tqdm

from src.frameworks import EdgeRAU
from src.run_helper import parse_args, save_results
from src.dataset_within_cluster import DM_Dataset_within_Cluster
from src.config import MODEL_DIR, RESULT_DIR, DATA_DIR
from forecast import ForecastNet

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
set_seed(521000)
props = parse_args(sys.argv[1:])
device = torch.device(f'cuda:{props.device_id}' if  torch.cuda.is_available() else 'cpu')
props.device = device

if props.dtype.lower() == "float32":
    props.dtype = torch.float32
elif props.dtype.lower() == "float16":
    props.dtype = torch.bfloat16
else:
    print("Only float32 and float16 are allowed")
    exit(1)

    # Define the loss
def loss_mlu(y_pred_batch, y_true_batch):
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        opt = y_true_batch[[i]]
        max_cong = torch.max(y_pred)
        loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong/max_cong.item()
        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt.item()
        losses.append(loss)
        loss_vals.append(loss_val)
    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    if ret_val < 0.99 and opt.item() != 100000:
        print("Loss value is less than 1.0")
        print(loss_vals)
        assert ret < 0.99

    return ret, ret_val

if props.mode == "train":
    initial_max_memory = torch.cuda.max_memory_allocated(device=device)
    model = EdgeRAU(props).to(device)

    train_ds_list = []
    train_dl_list = []
    for clust, start, end in zip(props.train_clusters, props.train_start_indices, props.train_end_indices):
        train_dataset = DM_Dataset_within_Cluster(props, clust, start, end)
        train_dl = DataLoader(train_dataset, batch_size=props.batch_size, shuffle=True)
        train_ds_list.append(train_dataset)
        train_dl_list.append(train_dl)

    num_paths = train_ds_list[0].tm_list[0].shape[0]
    forecast_model = ForecastNet(
        forecast_type=props.forecast_type,
        hist_len=props.hist_len,
        tm_shape=(num_paths,),
        hidden_dim=props.forecast_hidden_dim,
        hidden_layers=props.forecast_hidden_layers,
        alpha=props.forecast_alpha,
    ).to(device)
    forecast_optimizer = None
    if any(p.requires_grad for p in forecast_model.parameters()):
        forecast_optimizer = torch.optim.Adam(forecast_model.parameters(), lr=props.lr)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)

    val_ds_list = []
    val_dl_list = []
    for clust, start, end in zip(props.val_clusters, props.val_start_indices, props.val_end_indices):
        val_dataset = DM_Dataset_within_Cluster(props, clust, start, end)
        val_dl = DataLoader(val_dataset, batch_size=props.batch_size, shuffle=True)
        val_ds_list.append(val_dataset)
        val_dl_list.append(val_dl)

    forecast_trainable = forecast_optimizer is not None
    if forecast_trainable:
        total_epochs = props.epochs * 2
    else:
        total_epochs = props.epochs

    for epoch in range(total_epochs):
        train_edge = not forecast_trainable or epoch < props.epochs
        train_forecast = forecast_trainable and epoch >= props.epochs

        if train_edge:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            if forecast_trainable:
                forecast_model.eval()
                for param in forecast_model.parameters():
                    param.requires_grad = False
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            forecast_model.train()
            for param in forecast_model.parameters():
                param.requires_grad = True

        for i in range(len(train_ds_list)):
            train_dataset = train_ds_list[i]
            train_dl = train_dl_list[i]
            train_dataset.paths_to_edges = train_dataset.paths_to_edges.to(device, dtype=props.dtype)

            with tqdm(train_dl) as tepoch:
                loss_sum = loss_count = 0
                start_time = time()
                for i, inputs in enumerate(tepoch):
                    tepoch.set_description(f"Train Epoch {epoch+1}/{total_epochs}")
                    tms, tm_hist, capacities, opt = inputs

                    if not props.dynamic:
                        capacities = capacities[:1]

                    tms = tms.to(device, dtype=props.dtype)
                    tm_hist = tm_hist.to(device, dtype=props.dtype)
                    capacities = capacities.to(device, dtype=props.dtype)
                    opt = opt.to(device, dtype=props.dtype)

                    tms_pred = forecast_model(tm_hist)
                    if train_edge:
                        model_optimizer.zero_grad()
                    elif train_forecast:
                        forecast_optimizer.zero_grad()

                    preds = model(capacities, props, tms, tms_pred, tm_hist, train_dataset.paths_to_edges, props.num_for_loops)

                    loss, loss_val = loss_mlu(preds, opt)
                    loss.backward()

                    if train_edge:
                        model_optimizer.step()
                    elif train_forecast:
                        forecast_optimizer.step()

                    loss_sum += loss_val
                    loss_count += 1
                    tepoch.set_postfix(loss=loss_sum/loss_count)
                end_time = time()
                throughput = props.batch_size * len(train_dl) / (end_time - start_time)

            train_dataset.paths_to_edges = train_dataset.paths_to_edges.cpu()
        final_max_memory = torch.cuda.max_memory_allocated(device=device)
        used_memory = (final_max_memory - initial_max_memory) / 1024 ** 3
        

        model.eval()
        for i in range(len(val_ds_list)):
            val_dataset = val_ds_list[i]
            val_dl = val_dl_list[i]
            paths_to_edges = val_dataset.paths_to_edges.to(device, dtype=props.dtype)
            edges_to_paths = paths_to_edges.transpose(0, 1)
            with torch.no_grad():
                with tqdm(val_dl) as vepoch:
                    loss_sum = loss_count = 0
                    for i, inputs in enumerate(vepoch):
                        vepoch.set_description(f"Valid Epoch {epoch+1}/{props.epochs}")
                        tms, tm_hist, capacities, opt = inputs
                        if not props.dynamic:
                            capacities = capacities[:1]
                        tms = tms.to(device, dtype=props.dtype)
                        tm_hist = tm_hist.to(device, dtype=props.dtype)
                        capacities = capacities.to(device, dtype=props.dtype)
                        opt = opt.to(device, dtype=props.dtype)
                        tms_pred = forecast_model(tm_hist)
                        preds = model(capacities, props, tms, tms_pred, tm_hist, paths_to_edges, props.num_for_loops)
                        loss, loss_val = loss_mlu(preds, opt)
                        loss_sum += loss_val
                        loss_count += 1
                        vepoch.set_postfix(loss=loss_sum/loss_count)

        model_name = f"Model_{props.topo_name}_forecast_{props.forecast_type}_pred_{props.pred}_method_{props.pred_method}_{props.num_paths_per_pair}sp{'_' + props.model_name if props.model_name else ''}.pkl"
        model_path = os.path.join(MODEL_DIR, model_name)
        torch.save(model, model_path)

        forecast_name = f"Forecast_{props.forecast_type}_{props.topo_name}_{props.num_paths_per_pair}sp{'_' + props.model_name if props.model_name else ''}.pth"
        forecast_path = os.path.join(MODEL_DIR, forecast_name)
        torch.save(forecast_model.state_dict(), forecast_path)
    
    results_file = os.path.join(RESULT_DIR, "Train_results", "DualTE.csv")
    if not os.path.exists(os.path.dirname(results_file)):
        os.makedirs(os.path.dirname(results_file))
    results_data = {
        "topo_name": props.topo_name,
        "forecast_type": props.forecast_type,
        "num_paths_per_pair": props.num_paths_per_pair,
        "throughput": throughput,
        "memory_usage": used_memory,
        "train_clusters":'|'.join(map(str, props.train_clusters)),
    }
    save_results(results_data, results_file)

elif props.mode == "test":
    model_name = f"Model_{props.topo_name}_forecast_{props.forecast_type}_pred_{props.pred}_method_{props.pred_method}_{props.num_paths_per_pair}sp{'_' + props.model_name if props.model_name else ''}.pkl"
    model_path = os.path.join(MODEL_DIR, model_name)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(dtype=props.dtype)
    model.eval()

    parts = []
    if props.model_name:
        parts.append(props.model_name)
    elif props.pred:
        parts.append(f"DualTE_pred_{props.pred_method}")
    else:
        parts.append("DualTE")
    parts.append(f"forecast_{props.forecast_type}")
    results_filename = "_".join(parts) + "_results.csv"

    if props.failure_num != 0:
        results_filepath = os.path.join(RESULT_DIR, "Test_results", "DualTE", \
                                        props.topo_name, f'{props.num_paths_per_pair}sp', f'{props.test_cluster}',\
                                            f'failures_{props.failure_num}_{results_filename}')
    else:
        if props.pred:
            results_filepath = os.path.join(RESULT_DIR, "Test_pred_results", f"DualTE_pred_{props.pred_method}", \
                                    props.topo_name, f'{props.num_paths_per_pair}sp', f'{props.test_cluster}', \
                                        results_filename)
        else:
            results_filepath = os.path.join(RESULT_DIR, "Test_results", "DualTE", \
                                            props.topo_name, f'{props.num_paths_per_pair}sp', f'{props.test_cluster}', \
                                                results_filename)
    if not os.path.exists(os.path.dirname(results_filepath)):
        os.makedirs(os.path.dirname(results_filepath))

    match = re.match(r'^(.*?)_Split', props.topo_name)
    if match:
        props.topo_name = match.group(1)
    
    test_dataset = DM_Dataset_within_Cluster(props, props.test_cluster, props.test_start_idx, props.test_end_idx)
    num_paths = test_dataset.tm_list[0].shape[0]
    forecast_model = ForecastNet(
        forecast_type=props.forecast_type,
        hist_len=props.hist_len,
        tm_shape=(num_paths,),
        hidden_dim=props.forecast_hidden_dim,
        hidden_layers=props.forecast_hidden_layers,
        alpha=props.forecast_alpha,
    ).to(device)
    forecast_name = f"Forecast_{props.forecast_type}_{props.topo_name}_{props.num_paths_per_pair}sp{'_' + props.model_name if props.model_name else ''}.pth"
    forecast_path = os.path.join(MODEL_DIR, forecast_name)
    if os.path.exists(forecast_path):
        forecast_model.load_state_dict(torch.load(forecast_path, map_location=device, weights_only=False))
    else:
        print(f"[Warning] Forecast model file not found: {forecast_path}. Using default forecast weights.")
    forecast_model.eval()

    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_dataset.paths_to_edges = test_dataset.paths_to_edges.to(device, dtype=props.dtype)
    if props.failure_num:
        link_failures_path = os.path.join(DATA_DIR, props.topo_name,'link_failure',f"{props.failure_num}.npy")
        failure_links = np.load(link_failures_path)
    with torch.no_grad():
        with tqdm(test_dl) as tests:
            tests_losses = []
            for i, inputs in enumerate(tests):
                tms, tm_hist, capacities, opt = inputs
                tms = tms.to(device, dtype=props.dtype)
                tm_hist = tm_hist.to(device, dtype=props.dtype)
                capacities = capacities.to(device, dtype=props.dtype)
                opt = opt.to(device, dtype=props.dtype)
                if props.failure_num:
                    props.failures = failure_links[i].tolist()
                tms_pred = forecast_model(tm_hist)
                preds = model(capacities, props, tms, tms_pred, tm_hist, test_dataset.paths_to_edges, props.num_for_loops)
                loss, loss_val = loss_mlu(preds, opt)
                tests_losses.append(loss_val)
            print(f"Average loss: {sum(tests_losses)/len(tests_losses)}")

    with open(results_filepath, 'w') as file:
        for loss in tests_losses:
            file.write(f"{loss}\n")
        file.close()