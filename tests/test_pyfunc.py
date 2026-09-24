import numpy as np

import nutpie
from nutpie.compiled_pyfunc import from_pyfunc


def _compile_pyfunc_model():
    def make_logp_fn():
        def logp(point):
            return -0.5 * np.dot(point, point), -point

        return logp

    def make_expand_fn(seed1, seed2, chain):
        def expand(point):
            return {"x": np.asarray(point, dtype=np.float64)}

        return expand

    return from_pyfunc(
        ndim=2,
        make_logp_fn=make_logp_fn,
        make_expand_fn=make_expand_fn,
        expanded_dtypes=[np.float64],
        expanded_shapes=[(2,)],
        expanded_names=["x"],
    )


def test_pyfunc_with_init_point_fn_deterministic():
    compiled = _compile_pyfunc_model()
    expected = np.array([0.25, -0.75])
    seen = []

    def init_point_fn(rng):
        assert isinstance(rng, np.random.Generator)
        point = expected.copy()
        seen.append(point)
        return point

    compiled = compiled.with_init_point_fn(init_point_fn)
    nutpie.sample(
        compiled,
        chains=2,
        draws=5,
        tune=5,
        seed=1,
        progress_bar=False,
        return_raw_trace=True,
    )
    nutpie.sample(
        compiled,
        chains=2,
        draws=5,
        tune=5,
        seed=2,
        progress_bar=False,
        return_raw_trace=True,
    )

    assert len(seen) == 4
    for point in seen:
        np.testing.assert_allclose(point, expected)


def test_pyfunc_with_init_point_fn_stochastic():
    compiled = _compile_pyfunc_model()
    seen = []

    def init_point_fn(rng):
        assert isinstance(rng, np.random.Generator)
        point = rng.normal(size=2)
        assert np.isfinite(point).all()
        seen.append(point.copy())
        return point

    compiled = compiled.with_init_point_fn(init_point_fn)
    nutpie.sample(
        compiled,
        chains=1,
        draws=5,
        tune=5,
        seed=1,
        progress_bar=False,
        return_raw_trace=True,
    )
    nutpie.sample(
        compiled,
        chains=1,
        draws=5,
        tune=5,
        seed=2,
        progress_bar=False,
        return_raw_trace=True,
    )

    assert len(seen) == 2
    assert not np.allclose(seen[0], seen[1])


def test_pyfunc_with_init_point_fn_replacement():
    called = []

    def old_init_point_fn(rng):
        raise AssertionError("old init callback should have been replaced")

    def new_init_point_fn(rng):
        assert isinstance(rng, np.random.Generator)
        called.append(True)
        return np.array([0.25, -0.75])

    compiled = (
        _compile_pyfunc_model()
        .with_init_point_fn(old_init_point_fn)
        .with_init_point_fn(new_init_point_fn)
    )

    nutpie.sample(
        compiled,
        chains=1,
        draws=5,
        tune=5,
        seed=1,
        progress_bar=False,
        return_raw_trace=True,
    )

    assert called == [True]
