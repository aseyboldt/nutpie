import sys

import pytest

from nutpie.flow_adapter import make_transform_adapter


def test_rust_flow_needs_no_flowjax(monkeypatch):
    # As if the nnflow extras were not installed
    for name in ("flowjax", "equinox", "optax", "optimistix", "paramax"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "nutpie.transform_adapter", raising=False)

    adapter = make_transform_adapter()
    assert adapter.func.__name__ == "TriangularFlowAdapter"

    with pytest.raises(ImportError, match="flowjax"):
        make_transform_adapter(
            rust_flow=False, diag_restart_margin=None, scale_squash=1.0
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"window_size": 128}, {"window_size": 160, "val_fraction": 0.5}],
)
def test_window_too_small_to_train(kwargs):
    with pytest.raises(ValueError, match="draws to train on"):
        make_transform_adapter(**kwargs)


def test_window_large_enough():
    make_transform_adapter(window_size=192)
    make_transform_adapter(window_size=128, val_fraction=0.0)
