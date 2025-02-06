
```
python3 run_edgerau.py --topo_name GEANT \
--batch_size 16 --epochs 10 --lr 0.007 --dtype float32 \
--mode train --num_paths_per_pair 4 \
--num_path_mlp_hidden_layers 5 --num_edge_mlp_hidden_layers 1 --num_for_loop 8 \
--train_clusters 0 --train_start_indices 0 --train_end_indices 8080 \
--val_clusters 0 --val_start_indices 8080 --val_end_indices 10773 \
--device_id 1 --num_edge_mlp_hidden_dim 8
```

```
python3 run_edgerau.py --topo_name GEANT \
--batch_size 16 --epochs 10 --lr 0.007 --dtype float32 \
--mode test --num_paths_per_pair 4 \
--num_path_mlp_hidden_layers 5 --num_edge_mlp_hidden_layers 1 --num_for_loop 8 \
--test_cluster 0 --test_start_idx 8080 --test_end_idx 10773 \
--device_id 1
```