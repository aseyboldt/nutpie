import sys
import types

import numpy as np

from nutpie.compile_pymc import _make_initial_point_wrapper


class _FakeValueVar:
    def __init__(self, name):
        self.name = name


def test_make_initial_point_wrapper_uses_generator(monkeypatch):
    seeds = []

    def make_initial_point_fn(*, model, return_transformed):
        assert return_transformed is True

        def wrapped(seed):
            seeds.append(seed)
            base = int(seed % 10)
            return {
                "a": np.array(base, dtype=np.float64),
                "b": np.array([base + 1, base + 2], dtype=np.float64),
            }

        return wrapped

    fake_pymc = types.ModuleType("pymc")
    fake_initial_point = types.ModuleType("pymc.initial_point")
    fake_initial_point.make_initial_point_fn = make_initial_point_fn
    monkeypatch.setitem(sys.modules, "pymc", fake_pymc)
    monkeypatch.setitem(sys.modules, "pymc.initial_point", fake_initial_point)

    rv_a = object()
    rv_b = object()
    model = types.SimpleNamespace(
        free_RVs=[rv_a, rv_b],
        rvs_to_values={rv_a: _FakeValueVar("a"), rv_b: _FakeValueVar("b")},
    )

    init_point_fn = _make_initial_point_wrapper(model)
    first = init_point_fn(np.random.default_rng(1))
    second = init_point_fn(np.random.default_rng(2))

    assert len(seeds) == 2
    assert seeds[0] != seeds[1]
    np.testing.assert_allclose(
        first,
        np.array(
            [seeds[0] % 10, seeds[0] % 10 + 1, seeds[0] % 10 + 2],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        second,
        np.array(
            [seeds[1] % 10, seeds[1] % 10 + 1, seeds[1] % 10 + 2],
            dtype=np.float64,
        ),
    )
