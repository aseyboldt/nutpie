import numpy as np
import pytest

import nutpie


def _compile_stan_vector_model():
    model = """
    data {}
    parameters {
        vector[2] x;
    }
    model {
        x ~ normal(0, 1);
    }
    """

    return nutpie.compile_stan_model(code=model)


def test_stan_model():
    model = """
    data {}
    parameters {
        real a;
    }
    model {
        a ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    trace = nutpie.sample(compiled_model)
    trace.posterior.a  # noqa: B018


def test_stan_model_data():
    model = """
    data {
        real x;
    }
    parameters {
        real a;
    }
    model {
        a ~ normal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model)
    with pytest.raises(RuntimeError):
        trace = nutpie.sample(compiled_model)
    trace = nutpie.sample(compiled_model.with_data(x=np.array(3.0)))
    trace.posterior.a  # noqa: B018


def test_stan_with_init_point_fn_deterministic():
    compiled_model = _compile_stan_vector_model()
    expected = np.array([0.25, -0.75])
    seen = []

    def init_point_fn(rng):
        assert isinstance(rng, np.random.Generator)
        point = expected.copy()
        seen.append(point)
        return point

    compiled_model = compiled_model.with_init_point_fn(init_point_fn)
    nutpie.sample(
        compiled_model,
        chains=2,
        draws=5,
        tune=5,
        seed=1,
        progress_bar=False,
        return_raw_trace=True,
    )
    nutpie.sample(
        compiled_model,
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


def test_stan_with_init_point_fn_stochastic():
    compiled_model = _compile_stan_vector_model()
    seen = []

    def init_point_fn(rng):
        assert isinstance(rng, np.random.Generator)
        point = rng.normal(size=2)
        assert np.isfinite(point).all()
        seen.append(point.copy())
        return point

    compiled_model = compiled_model.with_init_point_fn(init_point_fn)
    nutpie.sample(
        compiled_model,
        chains=1,
        draws=5,
        tune=5,
        seed=1,
        progress_bar=False,
        return_raw_trace=True,
    )
    nutpie.sample(
        compiled_model,
        chains=1,
        draws=5,
        tune=5,
        seed=2,
        progress_bar=False,
        return_raw_trace=True,
    )

    assert len(seen) == 2
    assert not np.allclose(seen[0], seen[1])


def test_stan_with_init_point_fn_replacement():
    called = []

    def old_init_point_fn(rng):
        raise AssertionError("old init callback should have been replaced")

    def new_init_point_fn(rng):
        assert isinstance(rng, np.random.Generator)
        called.append(True)
        return np.array([0.25, -0.75])

    compiled_model = (
        _compile_stan_vector_model()
        .with_init_point_fn(old_init_point_fn)
        .with_init_point_fn(new_init_point_fn)
    )

    nutpie.sample(
        compiled_model,
        chains=1,
        draws=5,
        tune=5,
        seed=1,
        progress_bar=False,
        return_raw_trace=True,
    )

    assert called == [True]


def test_stan_param_roundtrip():
    model = """
    data {}
    parameters {
        real<lower=0> sigma;
    }
    model {
        sigma ~ lognormal(0, 1);
    }
    """

    compiled_model = nutpie.compile_stan_model(code=model).with_data()
    unconstrained = np.asarray(
        compiled_model.model.unconstrain_json('{"sigma": 1.5}'), dtype=np.float64
    )
    constrained = np.asarray(
        compiled_model.model.param_constrain(unconstrained), dtype=np.float64
    )
    unconstrained_roundtrip = np.asarray(
        compiled_model.model.unconstrain(constrained), dtype=np.float64
    )

    np.testing.assert_allclose(constrained, np.array([1.5]))
    np.testing.assert_allclose(unconstrained_roundtrip, unconstrained)
