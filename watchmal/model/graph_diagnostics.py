"""Opt-in graph-batch diagnostics for GAT forward passes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from statistics import median
from typing import Any

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class GraphDiagnosticsConfig:
    enabled: bool = False
    every: int = 0
    max_nodes: int = 0
    max_edges: int = 0
    validate_mode: str = "off"  # off|warn|raise
    log_cuda_memory: bool = True
    max_edge_node_ratio: float = 0.0

    @classmethod
    def from_env(cls) -> "GraphDiagnosticsConfig":
        mode = os.getenv("WATCHMAL_GRAPH_DIAGNOSTICS_VALIDATE", "off").strip().lower()
        if mode not in ("off", "warn", "raise"):
            mode = "off"
        return cls(
            enabled=_env_flag("WATCHMAL_GRAPH_DIAGNOSTICS", default=False),
            every=max(0, _env_int("WATCHMAL_GRAPH_DIAGNOSTICS_EVERY", default=0)),
            max_nodes=max(0, _env_int("WATCHMAL_GRAPH_DIAGNOSTICS_MAX_NODES", default=0)),
            max_edges=max(0, _env_int("WATCHMAL_GRAPH_DIAGNOSTICS_MAX_EDGES", default=0)),
            validate_mode=mode,
            log_cuda_memory=_env_flag("WATCHMAL_GRAPH_DIAGNOSTICS_LOG_CUDA_MEMORY", default=True),
            max_edge_node_ratio=max(0.0, _env_float("WATCHMAL_GRAPH_DIAGNOSTICS_MAX_EDGE_NODE_RATIO", default=0.0)),
        )

    def should_log(self, forward_counter: int, total_nodes: int, total_edges: int) -> bool:
        if not self.enabled:
            return False
        if self.every > 0 and forward_counter % self.every == 0:
            return True
        if self.max_nodes > 0 and total_nodes > self.max_nodes:
            return True
        if self.max_edges > 0 and total_edges > self.max_edges:
            return True
        return False


def _tensor_meta(tensor: Any) -> dict[str, Any] | None:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _cpu_long(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.long)


def _cpu_counts(tensor: torch.Tensor, n: int) -> list[int]:
    if n <= 0:
        return []
    return _cpu_long(torch.bincount(tensor, minlength=n)).tolist()


def event_ids_from_data(data: Any, num_graphs: int) -> tuple[list[int] | None, str]:
    idx = getattr(data, "idx", None)
    if idx is None:
        return None, "event IDs unavailable (batch has no data.idx)"
    if isinstance(idx, torch.Tensor):
        flat = idx.detach().view(-1)
        if flat.numel() == 0:
            return None, "event IDs unavailable (data.idx is empty)"
        values = [int(v) for v in flat.to(device="cpu", dtype=torch.long).tolist()]
    elif isinstance(idx, (list, tuple)):
        values = [int(v) for v in idx]
    else:
        values = [int(idx)]
    if num_graphs > 0 and len(values) != num_graphs:
        return values, f"event ID count mismatch: got {len(values)} IDs for {num_graphs} graph(s)"
    return values, "event IDs available"


def cuda_memory_snapshot(device: torch.device | None, enabled: bool = True) -> dict[str, Any] | None:
    if not enabled or device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None
    with torch.no_grad():
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
    return {
        "allocated_bytes": int(allocated),
        "reserved_bytes": int(reserved),
        "free_bytes": int(max(total - reserved, 0)),
        "total_bytes": int(total),
    }


def build_batch_summary(
    *,
    data: Any,
    batch: torch.Tensor,
    edge_index: torch.Tensor | None,
    num_graphs: int,
    forward_counter: int,
    stage: str,
    extra_tensors: dict[str, torch.Tensor] | None = None,
    include_cuda_memory: bool = True,
) -> dict[str, Any]:
    with torch.no_grad():
        node_counts = _cpu_counts(batch, num_graphs)
        total_nodes = int(batch.numel())

        edge_counts: list[int] = []
        total_edges = 0
        if edge_index is not None and edge_index.numel() > 0:
            row = edge_index[0]
            edge_counts = _cpu_counts(batch[row], num_graphs)
            total_edges = int(edge_index.size(1))

        event_ids, event_ids_status = event_ids_from_data(data, num_graphs)
        tensor_meta = {
            "batch": _tensor_meta(batch),
            "edge_index": _tensor_meta(edge_index),
            "data_x": _tensor_meta(getattr(data, "x", None)),
            "data_pos": _tensor_meta(getattr(data, "pos", None)),
        }
        for key, value in (extra_tensors or {}).items():
            tensor_meta[key] = _tensor_meta(value)

        node_stats = {
            "min": min(node_counts) if node_counts else 0,
            "max": max(node_counts) if node_counts else 0,
            "mean": (sum(node_counts) / len(node_counts)) if node_counts else 0.0,
            "median": float(median(node_counts)) if node_counts else 0.0,
        }
        edge_stats = {
            "min": min(edge_counts) if edge_counts else 0,
            "max": max(edge_counts) if edge_counts else 0,
            "mean": (sum(edge_counts) / len(edge_counts)) if edge_counts else 0.0,
        }
        return {
            "stage": stage,
            "forward_counter": int(forward_counter),
            "num_graphs": int(num_graphs),
            "total_nodes": total_nodes,
            "nodes_per_graph": node_counts,
            "nodes_per_graph_stats": node_stats,
            "total_edges": int(total_edges),
            "edges_per_graph": edge_counts,
            "edges_per_graph_stats": edge_stats,
            "event_ids": event_ids,
            "event_ids_status": event_ids_status,
            "tensor_meta": tensor_meta,
            "cuda_memory": cuda_memory_snapshot(batch.device, enabled=include_cuda_memory),
        }


def validate_graph_edges(
    *,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    allow_self_loops: bool,
    max_edge_node_ratio: float = 0.0,
) -> list[str]:
    issues: list[str] = []
    if edge_index.numel() == 0:
        return issues

    with torch.no_grad():
        row = _cpu_long(edge_index[0])
        col = _cpu_long(edge_index[1])
        batch_cpu = _cpu_long(batch)
        num_nodes = int(batch_cpu.numel())
        num_edges = int(row.numel())

        if int(row.min().item()) < 0 or int(col.min().item()) < 0:
            issues.append("edge_index has negative indices")
        if int(row.max().item()) >= num_nodes or int(col.max().item()) >= num_nodes:
            issues.append(
                f"edge_index has out-of-range indices (num_nodes={num_nodes}, "
                f"min={min(int(row.min()), int(col.min()))}, max={max(int(row.max()), int(col.max()))})"
            )

        valid_mask = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
        if valid_mask.any():
            row_valid = row[valid_mask]
            col_valid = col[valid_mask]
            cross_graph = batch_cpu[row_valid] != batch_cpu[col_valid]
            cross_count = int(cross_graph.sum().item())
            if cross_count > 0:
                issues.append(f"{cross_count} cross-graph edge(s) detected")

            if not allow_self_loops:
                self_loops = int((row_valid == col_valid).sum().item())
                if self_loops > 0:
                    issues.append(f"{self_loops} self-loop edge(s) detected while knn_loop is disabled")

        edge_keys = row * max(num_nodes, 1) + col
        unique_edges = int(torch.unique(edge_keys).numel())
        duplicates = num_edges - unique_edges
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate directed edge(s) detected")

        if max_edge_node_ratio > 0 and num_nodes > 0:
            ratio = num_edges / num_nodes
            if ratio > max_edge_node_ratio:
                issues.append(
                    f"edge/node ratio too high ({ratio:.3f} > {max_edge_node_ratio:.3f}); "
                    f"edges={num_edges}, nodes={num_nodes}, graphs={num_graphs}"
                )
    return issues
