# Custom initial points

Compiled PyMC, Stan, and pyfunc models can use a user-specified initialization
function instead of the backend's default. The callback receives a
`numpy.random.Generator` and must return a flat unconstrained parameter
vector:

```python
def init_point_fn(rng):
    return rng.normal(size=compiled.n_dim)

compiled = compiled.with_init_point_fn(init_point_fn)
trace = nutpie.sample(compiled)
```

If no callback is set, the previous backend-specific default initialization
is used unchanged.

A positional `chain: int` argument will be added to the callback signature
in a future release, once `nuts-rs` threads chain ids through
`Model::init_position`.

## Stan constrained/unconstrained helpers

For Stan models, nutpie also exposes helper methods for converting between
constrained and unconstrained parameterizations. This makes it possible to
build initial points from model-space values, including constrained
parameters:

```python
compiled = nutpie.compile_stan_model(code=code).with_data(mu=3.0)

def init_point_fn(rng):
    x_val = float(rng.normal())
    return compiled.model.unconstrain_json(f'{{"x": {x_val}}}')

compiled = compiled.with_init_point_fn(init_point_fn)
```

The Stan helpers available on `compiled.model` are `unconstrain`,
`unconstrain_json`, and `param_constrain`.
