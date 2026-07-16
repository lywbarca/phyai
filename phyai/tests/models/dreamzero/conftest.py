from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from phyai.layers.linear import _reset_for_test
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


def _fake_mesh(
    *,
    name: str = "model",
    sizes: dict[str, int] | None = None,
    ranks: dict[str, int] | None = None,
) -> Mesh:
    sizes = sizes or {}
    ranks = ranks or {}

    def size_of(axis: str) -> int:
        return sizes.get(axis, 1)

    def rank_of(axis: str) -> int:
        return ranks.get(axis, 0)

    tm = MagicMock()
    tm.mesh_dim_names = tuple(sizes.keys()) if sizes else ()
    names = tm.mesh_dim_names

    def size(axis):
        if isinstance(axis, str):
            return size_of(axis)
        return size_of(names[axis])

    tm.size.side_effect = size
    tm.get_local_rank.side_effect = rank_of
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    mesh = Mesh(tm, name=name)
    register_mesh(mesh)
    return mesh


@pytest.fixture
def fake_mesh():
    saved = dict(_meshes)
    try:
        yield _fake_mesh
    finally:
        _meshes.clear()
        _meshes.update(saved)
        _reset_for_test()
