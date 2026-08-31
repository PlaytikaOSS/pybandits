# MIT License
#
# Copyright (c) 2023 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Low-level NumPyro/forward primitives shared by the BNN and the Neural-Linear meta-model.

Backend-agnostic helpers live here so they can be reused without importing the heavy
``BaseBayesianNeuralNetwork`` class:

* :func:`forward_layers` — the layer-by-layer MLP forward pass (used by both the JAX/NumPyro
  training model and the NumPy sampling path).
* :func:`per_sample_linear` — the per-row linear layer for ``forward_layers`` when each row
  gathers its own weights (the joint cMAB engine's arm-indexing trick).
* :func:`run_svi_epochs` — the epoch-driver around an SVI training loop (restore-best-state,
  NaN guard, early stopping, progress bar). The caller owns building the ``SVI`` object and the
  per-step ``run_one_epoch`` closure, so the explicit-arg ``jax.jit`` pattern is preserved.
"""

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpyro.infer import SVI
from tqdm import trange

from pybandits.model.bnn._guide import _wrap_guide_with_kl_scale
from pybandits.model.bnn._typing import _Array


def forward_layers(
    *,
    next_layer_input: _Array,
    weights_biases: Sequence[tuple[_Array, _Array]],
    activation_fn: Callable[[_Array], _Array],
    linear_fn: Callable[[_Array, _Array, _Array], _Array],
    backend: ModuleType,
    use_residual_connections: bool,
) -> _Array:
    """Layer-by-layer forward computation for both JAX/NumPyro and NumPy backends.

    Per-layer ``(input_dim, output_dim)`` are read from ``w.shape[-2:]``, which holds for both
    the NumPyro weights ``(input_dim, output_dim)`` and the NumPy sampled weights
    ``(n_samples, input_dim, output_dim)``. The activation is applied to every layer except the
    last, whose raw linear output is returned (pre-sigmoid).

    Parameters
    ----------
    next_layer_input : _Array
        Network input, shape ``(batch, input_dim)``. May be a JAX or NumPy array.
    weights_biases : list[tuple[_Array, _Array]]
        Per-layer ``(weights, biases)``.
    activation_fn : Callable[[_Array], _Array]
        Activation function matching the backend.
    linear_fn : Callable[[_Array, _Array, _Array], _Array]
        Backend-specific linear transform ``(x, w, b) -> x @ w + b``.
    backend : ModuleType
        Array namespace — ``jnp`` (JAX) or ``np`` (NumPy); used for residual padding.
    use_residual_connections : bool
        Whether to add skip connections when ``output_dim >= input_dim``.

    Returns
    -------
    _Array
        The raw linear output of the final layer (pre-sigmoid).
    """
    n_layers = len(weights_biases)
    linear_transform = next_layer_input
    for layer_ind, (w, b) in enumerate(weights_biases):
        input_dim = int(w.shape[-2])
        output_dim = int(w.shape[-1])

        linear_transform = linear_fn(next_layer_input, w, b)

        if layer_ind < n_layers - 1:
            activated_output = activation_fn(linear_transform)
            # Add residual connection if enabled and dimensions allow
            if use_residual_connections and output_dim >= input_dim:
                if output_dim == input_dim:
                    next_layer_input = activated_output + next_layer_input
                else:
                    pad = backend.zeros((next_layer_input.shape[0], output_dim - input_dim))
                    next_layer_input = activated_output + backend.concatenate([next_layer_input, pad], axis=1)
            else:
                next_layer_input = activated_output

    return linear_transform


def per_sample_linear(t: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    """Per-sample linear layer for :func:`forward_layers`, used when each row has its own weights.

    Unlike the shared-weight case (``t @ w + b``, one ``(in, out)`` matrix for the whole batch), here
    every row gathers its own ``(in, out)`` weight matrix — e.g. the joint cMAB engine's arm-indexing
    trick, which gathers each row's arm's head weights before this call.

    Parameters
    ----------
    t : jax.Array of shape (batch, in)
        Per-row input.
    w : jax.Array of shape (batch, in, out)
        Per-row weight matrix.
    b : jax.Array of shape (batch, out)
        Per-row bias.

    Returns
    -------
    jax.Array of shape (batch, out)
        Per-row linear output.
    """
    return jnp.einsum("bi,bio->bo", t, w) + b


def run_svi_epochs(
    run_one_epoch: Callable[[Any, jax.Array], tuple[Any, jax.Array]],
    init_state: Any,
    epoch_factor_arrays: list[jax.Array],
    *,
    restore_best: bool = True,
    early_stopping_callback: Any | None = None,
    desc: str = "SVI",
) -> tuple[Any, np.ndarray]:
    """Drive an SVI training run epoch-by-epoch, returning ``(final_state, approx_history)``.

    The caller supplies ``run_one_epoch(svi_state, epoch_factors) -> (svi_state, per_step_losses)``
    — typically a ``jax.jit``-compiled ``jax.lax.scan`` over ``svi.update`` — and the initial SVI
    state. This function owns only the host-side epoch loop: averaging losses, the NaN-divergence
    guard, restore-best-state, optional early stopping, and the progress bar.

    Parameters
    ----------
    run_one_epoch : Callable[[Any, jax.Array], tuple[Any, jax.Array]]
        Runs one epoch of SVI steps; takes ``(svi_state, epoch_factors)`` and returns the updated
        state and the per-step loss array for that epoch.
    init_state : Any
        The initialized ``SVIState`` (from ``svi.init(...)``).
    epoch_factor_arrays : list[jax.Array]
        Per-epoch arrays of the per-step KL-annealing factor; one entry per epoch. Their lengths
        determine the number of SVI steps per epoch.
    restore_best : bool, optional
        If True, return the state with the lowest epoch-mean loss, by default True.
    early_stopping_callback : Any | None, optional
        Object exposing ``should_stop(loss) -> bool`` plus ``tolerance`` / ``diff_type`` /
        ``patience`` for logging. ``None`` disables early stopping. Note: the caller is
        responsible for ``reset()`` before invoking this function.
    desc : str, optional
        Progress-bar description, by default ``"SVI"``.

    Returns
    -------
    tuple[Any, np.ndarray]
        ``(final_state, approx_history)`` where *approx_history* is the concatenated per-step
        losses across all epochs that ran.
    """
    all_losses = []
    best_loss = float("inf")
    svi_state = init_state
    best_svi_state = svi_state
    n_epochs = len(epoch_factor_arrays)
    pbar = trange(n_epochs, desc=desc, leave=False)

    try:
        for epoch_idx, epoch_factors in enumerate(epoch_factor_arrays):
            svi_state, epoch_losses = run_one_epoch(svi_state, epoch_factors)

            epoch_np = np.array(epoch_losses)
            epoch_loss = float(np.mean(epoch_np))
            all_losses.append(epoch_np)
            pbar.update(1)
            pbar.set_postfix(loss=f"{epoch_loss:.4f}")

            if np.isnan(epoch_loss):
                raise ValueError(
                    f"SVI training diverged: loss is NaN at epoch {epoch_idx + 1}/{n_epochs}. "
                    "Consider reducing the learning rate or checking your data for invalid values."
                )

            if restore_best and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_svi_state = svi_state

            if early_stopping_callback is not None:
                if early_stopping_callback.should_stop(epoch_loss):
                    logger.info(
                        f"Early stopping at epoch {epoch_idx + 1}/{n_epochs}: "
                        f"loss change below {early_stopping_callback.tolerance} "
                        f"({early_stopping_callback.diff_type}) for "
                        f"{early_stopping_callback.patience} consecutive epochs. "
                        f"Best loss: {best_loss:.6f}, last loss: {epoch_loss:.6f}."
                    )
                    break
    finally:
        pbar.close()

    approx_history = np.concatenate(all_losses) if all_losses else np.array([])
    final_state = best_svi_state if restore_best else svi_state
    return final_state, approx_history


def run_svi(
    *,
    model: Callable,
    guide: Any,
    optimizer: Any,
    loss: Any,
    rng_key: jax.Array,
    model_args: tuple[Any, ...],
    epoch_factor_arrays: list[jax.Array],
    restore_best: bool = True,
    early_stopping_callback: Any | None = None,
    desc: str = "SVI",
) -> tuple[SVI, dict, np.ndarray, jax.Array]:
    """Build and drive an SVI training run around ``model``/``guide``, returning the fitted params.

    Wraps ``guide`` for symmetric KL annealing, constructs the :class:`SVI` object, initialises it,
    compiles one ``jax.lax.scan`` epoch body, and hands the host-side epoch loop to
    :func:`run_svi_epochs`. ``model_args`` are forwarded to ``svi.init``/``svi.update`` as **explicit**
    JIT arguments (never closure captures) so large data buffers stay out of the XLA constant pool.
    Shared by the single-arm BNN training loop and the joint multi-arm cMAB engine, which differ only
    in the ``model``/``guide`` they build and the ``model_args`` they feed.

    Parameters
    ----------
    model : Callable
        The NumPyro model, called as ``model(*model_args, kl_annealing_factor)``.
    guide : Any
        The (unwrapped) AutoGuide; wrapped here for KL annealing. The caller keeps the unwrapped
        reference for any downstream ``guide.median(...)`` extraction.
    optimizer : Any
        The optax/numpyro optimizer.
    loss : Any
        The ELBO loss instance (e.g. ``TraceMeanField_ELBO(num_particles=...)``).
    rng_key : jax.Array
        PRNG key; a fresh split is consumed for ``svi.init`` and the updated key is returned.
    model_args : tuple[Any, ...]
        Positional arguments threaded into ``model`` on every step (e.g. ``(x, y)`` or ``(x, arm_data)``).
    epoch_factor_arrays : list[jax.Array]
        Per-epoch arrays of the per-step KL-annealing factor.
    restore_best : bool, optional
        Return the lowest-loss state, by default True.
    early_stopping_callback : Any | None, optional
        Early-stopping callback (already ``reset()`` by the caller), by default None.
    desc : str, optional
        Progress-bar description, by default ``"SVI"``.

    Returns
    -------
    tuple[SVI, dict, np.ndarray, jax.Array]
        ``(svi, params, approx_history, rng_key)`` — the SVI object, the final variational params,
        the per-step loss history, and the advanced PRNG key.
    """
    scaled_guide = _wrap_guide_with_kl_scale(guide)
    svi = SVI(model, scaled_guide, optimizer, loss=loss)
    rng_key, subkey = jax.random.split(rng_key)
    svi_state = svi.init(subkey, *model_args, 1.0)

    # Pass model_args as explicit jit arguments (not closure captures) so JAX treats them as abstract
    # buffers rather than embedding them as XLA constants (which OOMs the compiler on large data).
    @jax.jit
    def _run_epoch(state: Any, margs: tuple[Any, ...], factors: jax.Array) -> tuple[Any, jax.Array]:
        def _body(s: Any, factor: jax.Array) -> tuple[Any, jax.Array]:
            return svi.update(s, *margs, factor)

        return jax.lax.scan(_body, state, factors)

    def _run_one_epoch(state: Any, epoch_factors: jax.Array) -> tuple[Any, jax.Array]:
        return _run_epoch(state, model_args, epoch_factors)

    final_state, approx_history = run_svi_epochs(
        _run_one_epoch,
        svi_state,
        epoch_factor_arrays,
        restore_best=restore_best,
        early_stopping_callback=early_stopping_callback,
        desc=desc,
    )
    return svi, svi.get_params(final_state), approx_history, rng_key
