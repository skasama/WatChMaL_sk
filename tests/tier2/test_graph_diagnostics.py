from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from watchmal.model.graph_diagnostics import (
    build_batch_summary,
    event_ids_from_data,
    validate_graph_edges,
)


def _batch_with_event_ids() -> Batch:
    g0 = Data(
        x=torch.randn(3, 2),
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    g0.idx = 101
    g1 = Data(
        x=torch.randn(2, 2),
        pos=torch.randn(2, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    g1.idx = 202
    g2 = Data(
        x=torch.randn(1, 2),
        pos=torch.randn(1, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    g2.idx = 303
    return Batch.from_data_list([g0, g1, g2])


@pytest.mark.graph
def test_batch_summary_reports_graph_stats_and_event_ids():
    batch = _batch_with_event_ids()
    summary = build_batch_summary(
        data=batch,
        batch=batch.batch,
        edge_index=batch.edge_index,
        num_graphs=batch.num_graphs,
        forward_counter=7,
        stage="pre_attention",
        include_cuda_memory=False,
    )

    assert summary["forward_counter"] == 7
    assert summary["num_graphs"] == 3
    assert summary["total_nodes"] == 6
    assert summary["nodes_per_graph"] == [3, 2, 1]
    assert summary["total_edges"] == 4
    assert summary["edges_per_graph"] == [2, 2, 0]
    assert summary["event_ids"] == [101, 202, 303]
    assert summary["event_ids_status"] == "event IDs available"
    assert summary["cuda_memory"] is None


@pytest.mark.graph
def test_event_ids_explicitly_report_missing_idx():
    batch = Batch.from_data_list([Data(x=torch.randn(2, 1)), Data(x=torch.randn(1, 1))])
    event_ids, status = event_ids_from_data(batch, num_graphs=2)
    assert event_ids is None
    assert "unavailable" in status


@pytest.mark.graph
def test_edge_validation_accepts_valid_graph():
    batch = _batch_with_event_ids()
    issues = validate_graph_edges(
        edge_index=batch.edge_index,
        batch=batch.batch,
        num_graphs=batch.num_graphs,
        allow_self_loops=True,
        max_edge_node_ratio=2.0,
    )
    assert issues == []


@pytest.mark.graph
def test_edge_validation_flags_invalid_indices_and_cross_graph_edges():
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 4, 3, 2]], dtype=torch.long)
    issues = validate_graph_edges(
        edge_index=edge_index,
        batch=batch,
        num_graphs=2,
        allow_self_loops=True,
    )
    assert any("out-of-range" in issue for issue in issues)
    assert any("cross-graph" in issue for issue in issues)


@pytest.mark.graph
def test_edge_validation_flags_duplicates_and_self_loops():
    batch = torch.tensor([0, 0, 0], dtype=torch.long)
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=torch.long)
    issues = validate_graph_edges(
        edge_index=edge_index,
        batch=batch,
        num_graphs=1,
        allow_self_loops=False,
    )
    assert any("duplicate" in issue for issue in issues)
    assert any("self-loop" in issue for issue in issues)
