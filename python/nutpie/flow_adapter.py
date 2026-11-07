"""Build the flow adapter for ``adaptation="flow"`` from its settings.

`make_transform_adapter` is the only entry point the compiled models need. The
default adapter (``rust_flow=True``) is `triangular_adapter.TriangularFlowAdapter`,
which needs neither JAX nor flowjax; the JAX implementation in `transform_adapter`
is imported only for ``rust_flow=False``.
"""

import math
from functools import partial

import numpy as np

from nutpie._lib import FisherResiduals
from nutpie.triangular_adapter import MIN_TRAIN_DRAWS, _select_draws


def make_transform_adapter(
    *,
    verbose=False,
    window_size=512,
    show_progress=False,
    nn_depth=None,
    nn_width=4,
    num_layers=8,
    num_diag_windows=9,
    learning_rate=5e-4,
    untransformed_dim=None,
    zero_init=True,
    batch_size=128,
    reuse_opt_state=False,
    max_patience=20,
    householder_layer=False,
    dct_layer=False,
    gamma=None,
    log_inside_batch=False,
    fisher_ema_alpha=0.1,
    initial_skip=120,
    extension_windows=None,
    extend_dct=False,
    extension_var_count=4,
    extension_var_trafo_count=2,
    debug_save_bijection=False,
    make_optimizer=None,
    coupling_type="triangular",
    mvscale_layer=False,
    num_project=None,
    num_embed=None,
    num_householder=8,
    twin_layers=False,
    activation=None,
    max_epochs=60,
    transformer="asymmetric_sas",
    transformer_layers=1,
    tangent_sas_fix_b=False,
    log_gamma_bounds=None,
    scale_squash=8.0,
    input_squash=None,
    reuse_embed=True,
    order=None,
    sparsity=None,
    location_skip=True,
    scale_skip=True,
    feature_degree=None,
    method="lm-rust",
    solver_rtol=5e-2,
    solver_atol=1e-6,
    lm_linear_steps=80,
    lm_min_loss=math.exp(-3),  # noqa: B008
    lm_probe_batch=32,
    lm_probes=1024,
    lm_residual_batch=128,
    lm_cholesky_jitter=None,
    lm_mlp_ridge=1.0,
    lm_skip_ridge=0.1,
    lm_scale_skip_ridge=1.0,
    lm_shape_ridge=10.0,
    lm_probe_groups=None,
    lm_probe_rounds=1,
    lm_fit_affine=False,
    lm_patience=5,
    lm_line_search=True,
    lm_forcing="model",
    lm_exact_blocks=True,
    lm_max_exact_block_size=256,
    lm_lmp_size=8,
    lm_lmp_tol=1e-8,
    lm_preconditioner="blocks",
    lm_diagonal_probes=1,
    native_flow=True,
    stop_event=None,
    forget_fraction=0.5,
    recency=1.0,
    val_fraction=0.3,
    val_block_size=16,
    early_stopping=True,
    linear_fits=1,
    refit_diag=False,
    diag_restart_margin=0.0,
    rust_flow=True,
    conditioners=None,
):
    """The adapter for `nutpie.sample`'s ``transform_adapt``.

    With ``rust_flow``, the triangular flow is built, fitted and handed to the
    sampler in Rust, without a JAX flow (see
    `nutpie.triangular_adapter.TriangularFlowAdapter`). That needs
    ``coupling_type="triangular"`` and ``method="lm-rust"``, and supports the
    settings that apply to them; the others raise.

    Draw selection for each flow fit: draws before ``initial_skip`` and the
    oldest ``forget_fraction`` of all draws are ignored, the rest is thinned
    to at most ``window_size`` (never upsampled), and a random
    ``val_fraction`` of blocks of ``val_block_size`` consecutive draws is
    held out. Thinning keeps a density of draws that grows like
    ``t**recency`` over the remaining range, ``t`` from 0 at its oldest to 1
    at its newest draw: ``recency=0`` thins evenly, the default ``1`` ramps
    up linearly, since the draws usually get better as the flow does.

    The held-out draws decide whether a refit replaces the flow and, with
    ``early_stopping`` and ``method="lm-rust"``, when the LM fit stops (it
    then returns the step with the best validation loss).
    ``early_stopping=False`` still reports the validation loss but lets the
    fit run to its training-loss stopping rule. ``val_fraction=0`` trains on
    everything and compares flows on the training draws.

    ``transformer`` is the elementwise transformer of every conditional,
    ``"affine"``, ``"sas"``, ``"tangent_sas"``, ``"asymmetric_sas"`` or
    ``"asymmetric_affine"``, with ``transformer_layers`` layers of it; see
    `normalizing_flow.make_transformer` for how each one stacks.

    ``log_gamma_bounds`` bounds each ``"sas"`` or ``"asymmetric_sas"``
    layer's tail exponent, ``log gamma``. ``gamma`` is an exponent (``T(x) ~ |x|**gamma``), so
    without a bound a conditioner whose output grows linearly away from the
    training draws makes the map blow up polynomially with a growing power
    there; ``(-1, 1)`` allows tails at most ``e`` times heavier or lighter
    per layer. ``None`` leaves it unbounded.

    ``scale_squash`` (with ``rust_flow``) is the width ``c`` of the map from
    the transformer's scale field ``s`` to its log scale, ``c * asinh(s /
    c)``: linear within about ``c`` of zero, logarithmic beyond, so that a
    scale growing linearly in the parents away from the draws gives a scale
    growing only polynomially there. ``None`` makes the log scale ``s``
    itself, so that with ``scale_skip`` it is linear in the parents, as in a
    funnel (``x ~ N(0, exp(v / 2))``), which the default ``1`` cannot
    represent beyond a narrow range of ``v``.

    ``input_squash`` makes the conditioner MLPs of a triangular flow see
    ``c * asinh(y / c)`` of each parent, so that away from the draws their
    outputs grow like ``log`` of the distance instead of linearly; the
    location skip stays linear. `Contract2`'s ``beta`` and ``sigma`` act on
    the map through a power ``1 / gamma``, so with raw parents their linear
    growth away from the draws becomes a high-power blowup of the map.
    ``None`` gives the MLPs the raw parents.

    With ``method="lm-rust"``, the first ``linear_fits`` fits of a new flow
    fit only what keeps its conditionals Gaussian. With ``rust_flow``, these
    fits use the linear part of the conditioners alone
    (`nutpie.triangular_flow.linear_conditioners`: the constant off the shape
    coefficients and the linear conditioner on the location), and later
    fits add the rest, the MLPs starting near zero; otherwise they hold the
    rest fixed (see `fit_to_data`'s ``lm_linear_fit``). These fit the flow
    without MLPs and with affine transformers, which converges quickly;
    later fits start from it and use the MLPs and the shape coefficients
    where the draws support them. A refit only replaces the flow if it does
    better on the held-out draws, so the MLPs cannot make it worse there
    than this linear start, up to the noise of those draws.

    With ``refit_diag`` (and ``rust_flow``), every refit first replaces the
    flow's diagonal affine by one fitted to that window's training draws,
    as a new flow gets, and then fits the conditioners from their current
    parameters in those new coordinates (with the location's constant and
    linear weights, and the scale's constant, moved to keep their meaning). Otherwise the affine stays the one
    the flow was created with, from the draws of its first window.

    With ``diag_restart_margin`` (and ``rust_flow``), a refit starts from a
    new flow, fitted as the first one, instead of the current flow if a
    diagonal affine fitted to the window's draws beats the current flow by
    more than ``diag_restart_margin`` in ``log F`` on the held-out draws:
    then the flow's parameters from earlier windows do more harm than good,
    for instance if the chain only reached the typical set after the flow
    was built. The new flow replaces the current one only if it does better.
    A negative margin also restarts flows the diagonal comes close to.
    Flows that have only had their ``linear_fits`` are not restarted: they
    do about as well as the diagonal by construction. ``None`` never
    restarts.

    The LM fit's ridges (see `fit_to_data`): ``lm_mlp_ridge`` penalizes the
    conditioner MLPs' weights, with the flow without MLPs as its minimum;
    ``lm_skip_ridge`` the linear location skip, which needs it where a
    variable has about as many parents as there are draws; and
    ``lm_shape_ridge`` additionally what feeds the transformer's shape
    fields, pulling each conditional towards an affine transformer.
    ``scale_skip`` adds a second linear skip, from the parents to the
    transformer's scale field (in the transformer's own parameterization of
    it), regularized by ``lm_scale_skip_ridge`` and held at zero by the
    linear fits. What
    moves a variable's location (the skip and the MLP's output weights into
    it) is measured against the conditional's width, refreshed every LM
    step. All are ridges on the sum over the draws, so they weaken as the
    windows grow.

    With ``rust_flow``, ``conditioners`` (a list of ``{"kind": ...}``
    dicts, see `nutpie.triangular_flow.default_conditioners`) sets each
    variable's conditioners and their ridges directly; then ``nn_width``,
    ``activation``, ``input_squash``, ``scale_skip``, ``zero_init`` and the
    ``lm_*_ridge`` arguments are not used.

    ``lm_preconditioner`` (with ``rust_flow``) is what the LM fit's CG
    preconditioner and Marquardt diagonal come from: ``"blocks"``, the exact
    Gauss-Newton blocks (split at ``lm_max_exact_block_size``), or
    ``"diagonal_estimate"``, a Monte Carlo estimate of the diagonal of
    ``J^T J`` from ``lm_diagonal_probes`` random-sign pullbacks. Each probe
    costs about half a CG iteration and needs no selected inverse, so the
    estimate is much cheaper than the blocks for large conditioners, but it
    ignores the correlations within a conditioner and is noisy, about
    ``sqrt(2 / n_draws)`` relative with one probe.
    """
    if refit_diag and not rust_flow:
        raise ValueError("refit_diag is only supported with rust_flow=True.")
    if diag_restart_margin is not None and not rust_flow:
        raise ValueError("diag_restart_margin is only supported with rust_flow=True.")
    if lm_preconditioner != "blocks" and not rust_flow:
        raise ValueError("lm_preconditioner is only supported with rust_flow=True.")
    if scale_squash != 1.0 and not rust_flow:
        raise ValueError("scale_squash is only supported with rust_flow=True.")
    # The adapter keeps the diagonal preconditioner while a window leaves fewer than
    # MIN_TRAIN_DRAWS training draws after the validation split, so a window that
    # is too small would never fit a flow. Catch that up front.
    train, _ = _select_draws(
        window_size,
        0,
        max_draws=window_size,
        recency=recency,
        val_fraction=val_fraction,
        block_size=val_block_size,
        multiple=FisherResiduals.simd_width(),
        rng=np.random.default_rng(0),
    )
    if len(train) < MIN_TRAIN_DRAWS:
        raise ValueError(
            f"window_size={window_size} with val_fraction={val_fraction} leaves "
            f"{len(train)} draws to train on, but a flow fit needs at least "
            f"{MIN_TRAIN_DRAWS}, so the diagonal preconditioner would be kept "
            "forever. Use a larger window_size or a smaller val_fraction."
        )
    if conditioners is not None and not rust_flow:
        raise ValueError("conditioners is only supported with rust_flow=True.")
    if rust_flow:
        from nutpie.triangular_adapter import TriangularFlowAdapter

        unsupported = {
            "coupling_type": coupling_type != "triangular",
            "method": method != "lm-rust",
            "nn_depth": nn_depth not in (None, 1),
            "location_skip": not location_skip,
            "feature_degree": feature_degree is not None,
            "transformer": transformer == "asymmetric_affine",
            "extension_windows": bool(extension_windows),
        }
        unsupported = [name for name, bad in unsupported.items() if bad]
        if unsupported:
            raise ValueError(
                "rust_flow=True needs coupling_type='triangular', "
                "method='lm-rust', depth-1 conditioners with the location skip, "
                "a transformer other than 'asymmetric_affine', and no parent "
                f"features or extension windows; got {', '.join(unsupported)}."
            )
        return partial(
            TriangularFlowAdapter,
            sparsity=sparsity,
            order=order,
            verbose=verbose,
            window_size=window_size,
            num_diag_windows=num_diag_windows,
            initial_skip=initial_skip,
            forget_fraction=forget_fraction,
            recency=recency,
            val_fraction=val_fraction,
            val_block_size=val_block_size,
            early_stopping=early_stopping,
            linear_fits=linear_fits,
            refit_diag=refit_diag,
            diag_restart_margin=diag_restart_margin,
            debug_save_bijection=debug_save_bijection,
            stop_event=stop_event,
            nn_width=nn_width,
            activation="softplus" if activation is None else activation,
            zero_init=zero_init,
            transformer=transformer,
            transformer_layers=transformer_layers,
            tangent_sas_fix_b=tangent_sas_fix_b,
            log_gamma_bounds=log_gamma_bounds,
            scale_squash=scale_squash,
            input_squash=input_squash,
            scale_skip=scale_skip,
            max_epochs=max_epochs,
            solver_rtol=solver_rtol,
            lm_linear_steps=lm_linear_steps,
            lm_min_loss=lm_min_loss,
            lm_mlp_ridge=lm_mlp_ridge,
            lm_skip_ridge=lm_skip_ridge,
            lm_scale_skip_ridge=lm_scale_skip_ridge,
            lm_shape_ridge=lm_shape_ridge,
            lm_patience=lm_patience,
            lm_forcing=lm_forcing,
            lm_max_exact_block_size=lm_max_exact_block_size,
            lm_lmp_size=lm_lmp_size,
            lm_lmp_tol=lm_lmp_tol,
            lm_preconditioner=lm_preconditioner,
            lm_diagonal_probes=lm_diagonal_probes,
            conditioners=conditioners,
        )

    # Only the JAX flow needs JAX, flowjax and the rest of the nnflow extras.
    import jax

    from nutpie.transform_adapter import TransformAdapter, make_flow

    if extension_windows is None:
        extension_windows = []
    if activation is None and coupling_type == "triangular":
        # The Rust LM fit (`method="lm-rust"`) supports softplus and GELU
        # (`activation="gelu"`).
        activation = jax.nn.softplus

    return partial(
        TransformAdapter,
        verbose=verbose,
        window_size=window_size,
        make_flow_fn=partial(
            make_flow,
            householder_layer=householder_layer,
            dct_layer=dct_layer,
            nn_depth=nn_depth,
            nn_width=nn_width,
            n_embed=num_project,
            n_deembed=num_embed,
            mvscale=mvscale_layer,
            kind=coupling_type,
            num_householder=num_householder,
            twin_layers=twin_layers,
            activation=activation,
            transformer=transformer,
            transformer_layers=transformer_layers,
            tangent_sas_fix_b=tangent_sas_fix_b,
            log_gamma_bounds=log_gamma_bounds,
            input_squash=input_squash,
            reuse_embed=reuse_embed,
            order=order,
            sparsity=sparsity,
            location_skip=location_skip,
            scale_skip=scale_skip,
            feature_degree=feature_degree,
        ),
        show_progress=show_progress,
        num_diag_windows=num_diag_windows,
        learning_rate=learning_rate,
        zero_init=zero_init,
        untransformed_dim=untransformed_dim,
        batch_size=batch_size,
        reuse_opt_state=reuse_opt_state,
        max_patience=max_patience,
        gamma=gamma,
        log_inside_batch=log_inside_batch,
        fisher_ema_alpha=fisher_ema_alpha,
        initial_skip=initial_skip,
        extension_windows=extension_windows,
        extend_dct=extend_dct,
        extension_var_count=extension_var_count,
        extension_var_trafo_count=extension_var_trafo_count,
        debug_save_bijection=debug_save_bijection,
        make_optimizer=make_optimizer,
        num_layers=num_layers,
        max_epochs=max_epochs,
        method=method,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        lm_linear_steps=lm_linear_steps,
        lm_min_loss=lm_min_loss,
        lm_probe_batch=lm_probe_batch,
        lm_probes=lm_probes,
        lm_residual_batch=lm_residual_batch,
        lm_cholesky_jitter=lm_cholesky_jitter,
        lm_mlp_ridge=lm_mlp_ridge,
        lm_skip_ridge=lm_skip_ridge,
        lm_scale_skip_ridge=lm_scale_skip_ridge,
        lm_shape_ridge=lm_shape_ridge,
        lm_probe_groups=lm_probe_groups,
        lm_probe_rounds=lm_probe_rounds,
        lm_fit_affine=lm_fit_affine,
        lm_patience=lm_patience,
        lm_line_search=lm_line_search,
        lm_forcing=lm_forcing,
        lm_exact_blocks=lm_exact_blocks,
        lm_max_exact_block_size=lm_max_exact_block_size,
        lm_lmp_size=lm_lmp_size,
        lm_lmp_tol=lm_lmp_tol,
        native_flow=native_flow,
        stop_event=stop_event,
        forget_fraction=forget_fraction,
        recency=recency,
        val_fraction=val_fraction,
        val_block_size=val_block_size,
        early_stopping=early_stopping,
        linear_fits=linear_fits,
    )
