from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from critics_eye.utils.data_prep import collect_line_theme_edges


def test_collect_line_theme_edges_can_select_membership_relation() -> None:
	data = HeteroData()
	data[("line", "reflects_theme", "theme")].edge_index = torch.tensor(
		[[0, 1], [3, 4]]
	)
	data[("line", "symbolize", "theme")].edge_index = torch.tensor([[2], [5]])

	edge_types = set(data.edge_types)
	assert collect_line_theme_edges(data).shape[1] == 3
	filtered = collect_line_theme_edges(data, membership_relation="reflects_theme")
	assert torch.equal(filtered, torch.tensor([[0, 1], [3, 4]]))
	assert set(data.edge_types) == edge_types

	with pytest.raises(ValueError, match="missing"):
		collect_line_theme_edges(data, membership_relation="missing")
