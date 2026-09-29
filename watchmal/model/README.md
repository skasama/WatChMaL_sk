# Models

Every model is a plain PyTorch `nn.Module`, selected from the config via `_target_`
(see the [configuration guide](../../README.md#part-2--configuration-guide)). The model
never trains itself — the **engine** (`watchmal/engine/`) drives it, so a model and an
engine are always picked as a pair in the main config.

## Graph models (PyTorch Geometric)

| File | Class | Task | Shipped config |
|---|---|---|---|
| `gcns.py` | `BaseGCN` | graph classification | `model/gcn_classifier.yaml` |
| `gat.py` | `GraphAttentionNetwork` | graph classification / regression | `model/vanilla_gat_classifier.yaml`, `model/gat_cls_classifier.yaml`, `model/vanilla_gat_vertex_regressor.yaml` |
| `no_conv_mlp.py` | `NoConvMLP` | MLP baseline (no message passing) | — |
| `node_encoder.py` | `NodeEncoder` | shared node-embedding building block, and `get_activation` | — |
| `knn_edges.py` | `build_knn_edge_index` | k-nearest-neighbour edges built inside a forward pass | — |
| `mPMT_gat_pooling.py` | `HierarchicalGAT` | two-level PMT → mPMT GAT (WCTE) | `model/mpmt_gat_pooling.yaml` |
| `mPMT_gat_augmem.py` | `HierarchicalGAT` | memory-augmented variant of the above | — |

`GraphAttentionNetwork` covers both readouts. `num_cls_tokens: 0` averages the hit
representations; `num_cls_tokens: 1` or more appends virtual classification nodes joined
to every hit of their own event and reads the event representation off them. Setting
`knn_k` moves edge construction into the forward pass, from `data.pos`, which requires a
dataset that stores hit coordinates there — the stored PyG datasets do not.

### GAT graph diagnostics (opt-in, default off)

`watchmal/model/gat.py` has optional diagnostics for investigating pathological batches
and CUDA OOM failures without changing the model math.

- `WATCHMAL_GRAPH_DIAGNOSTICS=1`: enable diagnostics.
- `WATCHMAL_GRAPH_DIAGNOSTICS_EVERY=N`: periodic logging every `N` model forwards.
- `WATCHMAL_GRAPH_DIAGNOSTICS_MAX_NODES=N`: force a detailed log when a batch exceeds `N` nodes.
- `WATCHMAL_GRAPH_DIAGNOSTICS_MAX_EDGES=N`: force a detailed log when a batch exceeds `N` edges.
- `WATCHMAL_GRAPH_DIAGNOSTICS_VALIDATE=off|warn|raise`: edge validation mode.
- `WATCHMAL_GRAPH_DIAGNOSTICS_MAX_EDGE_NODE_RATIO=R`: validation threshold for edges/nodes.
- `WATCHMAL_GRAPH_DIAGNOSTICS_LOG_CUDA_MEMORY=0|1`: include CUDA allocated/reserved/free memory.

Example run:

```bash
WATCHMAL_GRAPH_DIAGNOSTICS=1 \
WATCHMAL_GRAPH_DIAGNOSTICS_EVERY=250 \
WATCHMAL_GRAPH_DIAGNOSTICS_MAX_NODES=10000 \
WATCHMAL_GRAPH_DIAGNOSTICS_MAX_EDGES=100000 \
WATCHMAL_GRAPH_DIAGNOSTICS_VALIDATE=warn \
WATCHMAL_GRAPH_DIAGNOSTICS_MAX_EDGE_NODE_RATIO=12 \
WATCHMAL_GRAPH_DIAGNOSTICS_LOG_CUDA_MEMORY=1 \
python main.py --config-path tutorial/config/caverns/main --config-name gat_vertex_regression
```

Interpretation rule of thumb:

- **Pathological event/graph**: one batch shows a sharp spike in per-graph nodes/edges
  (and often one event ID dominates the max).
- **Cumulative growth/leak-like behaviour**: per-graph sizes stay normal but CUDA
  allocated/reserved trend upward across many forwards until OOM.

Pair these with the graph engines: `engine/gnn_classifier.yaml`
(→ `watchmal.engine.graph.classification.ClassifierEngine`) or `engine/gnn_regressor.yaml`
(→ `watchmal.engine.graph.regression.RegressionEngine`). Models without a shipped config
are research models — read their docstrings and write a small `model/*.yaml` for them.

## Multi-ring segmentation (`multiring/`)

A sparse 3D CNN assembled from three parts by `watchmal.utils.build_utils.build_segmentation_model`
(see `model/multiring/segmentation_model.yaml` for the wiring):

- **Encoder** — `encoders/sparse_unet3D.SparseUNet3D`: sparse 3D U-Net (requires `spconv`).
- **Head** — `heads/query_transformer.QueryPerVoxelSoftmaxHead`: query-transformer decoder
  producing per-voxel ring assignments. `heads/per_voxel_linear.PerVoxelLinearHead` is a
  simpler alternative; `heads/activation_functions.py` provides softmax / sparsemax / entmax.
- **Wrapper** — `wrappers/model_wrapper.MultiRingModel`: glues encoder + head.

Pair with `engine/multiring/segmentation.yaml`
(→ `watchmal.engine.multiring.segmentation.MultiRingSegEngine`).

## Adding your own model

1. Create `watchmal/model/my_model.py` with an `nn.Module` whose `__init__` takes your
   hyperparameters.
2. Add a config in *your* workspace, `config/model/my_model.yaml`:
   ```yaml
   _target_: watchmal.model.my_model.MyModel
   in_channels: 5
   hidden_dim: 64
   ```
3. Reference it in your main config's `defaults` list: `- model: my_model`.
4. Keep the engine's contract in mind: your `forward` must accept the batches produced by
   the dataset/loader and return what the chosen engine expects — mirror a shipped model
   for the same task when in doubt.
