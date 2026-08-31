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
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as npdist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.util import helpful_support_errors


class ParameterizedScaleAutoNormal(AutoNormal):
    """AutoNormal guide with per-site scale initialization via a callable.

    Extends :class:`~numpyro.infer.autoguide.AutoNormal` so that the initial
    variational scale of every latent site can be set individually rather than
    sharing a single global scalar.  All other behaviour (diagonal-normal
    approximation, ``init_loc_fn``, plate handling, constrained-support
    transforms) is inherited unchanged from the parent.

    Two modes are supported:

    * **Per-site** (``init_scale_fn`` provided): the callable is invoked once
      per latent site and must return a value broadcast-compatible with that
      site's shape.  ``init_scale`` is ignored in this mode.
    * **Scalar fallback** (``init_scale_fn=None``): behaves identically to
      :class:`~numpyro.infer.autoguide.AutoNormal` with the given scalar
      ``init_scale``.

    Parameters
    ----------
    model : callable
        The NumPyro model whose posterior is being approximated.
    prefix : str, optional
        String prefix prepended to all variational parameter names, by default
        ``"auto"``.
    init_loc_fn : callable, optional
        Per-site initialisation strategy for the location parameters; see
        :mod:`numpyro.infer.initialization`.  Defaults to
        :func:`~numpyro.infer.initialization.init_to_uniform`.
    init_scale : float, optional
        Global initial scale used when ``init_scale_fn`` is ``None``, by
        default ``0.1``.
    init_scale_fn : Callable[[str], float | np.ndarray | jax.Array] | None, optional
        Callable with signature ``(site_name: str) -> scale`` where ``scale``
        is a scalar or an array broadcast-compatible with the site's shape.
        When provided, ``init_scale`` is ignored.  ``None`` (default) falls
        back to the scalar ``init_scale``.
    create_plates : callable, optional
        Optional function that creates :func:`numpyro.plate` contexts; passed
        directly to the parent ``AutoNormal.__init__``.

    Raises
    ------
    ValueError
        If ``init_scale_fn`` is provided but is not callable.

    Examples
    --------
    Seed each BNN weight/bias site from the current posterior sigma stored in
    ``model_params``::

        site_sigmas = {
            "weight_0": layer_params.weight.params["sigma"],
            "bias_0":   layer_params.bias.params["sigma"],
        }
        guide = ParameterizedScaleAutoNormal(
            numpyro_model,
            init_loc_fn=init_to_value(values={"weight_0": w_mu, "bias_0": b_mu}),
            init_scale_fn=lambda name: site_sigmas.get(name, fallback_sigma),
        )
    """

    def __init__(
        self,
        model: Callable,
        *,
        prefix: str = "auto",
        init_loc_fn: Callable = init_to_uniform,
        init_scale: float = 0.1,
        init_scale_fn: Callable[[str], float | np.ndarray | jax.Array] | None = None,
        create_plates: Callable | None = None,
    ) -> None:
        if init_scale_fn is not None and not callable(init_scale_fn):
            raise ValueError("init_scale_fn must be callable or None.")
        self._init_scale_fn = init_scale_fn
        super().__init__(
            model,
            prefix=prefix,
            init_loc_fn=init_loc_fn,
            init_scale=init_scale,
            create_plates=create_plates,
        )

    def _get_site_init_scale(self, name: str, init_loc: jax.Array) -> jax.Array:
        """Return the initial scale for ``name``, broadcast to match ``init_loc``.

        Parameters
        ----------
        name : str
            NumPyro site name.
        init_loc : jax.Array
            Initialised location array for this site; used only to determine
            the required output shape.

        Returns
        -------
        jax.Array
            Scale array with the same shape as ``init_loc``.
        """
        if self._init_scale_fn is not None:
            raw = self._init_scale_fn(name)
            return jnp.broadcast_to(jnp.array(raw), jnp.shape(init_loc))
        return jnp.full(jnp.shape(init_loc), self._init_scale)

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, jax.Array]:
        """Sample the variational posterior and register variational parameters.

        On the first call the prototype trace is built via
        ``_setup_prototype``; subsequent calls reuse it.  For every unobserved
        sample site a ``Normal(loc, scale)`` variational distribution is
        registered via :func:`numpyro.param`.  The initial scale is provided
        by :meth:`_get_site_init_scale`; for constrained sites the Normal is
        wrapped in a :class:`~numpyro.distributions.TransformedDistribution`
        using the site's bijection.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the model (used only during
            prototype construction).
        **kwargs : Any
            Keyword arguments forwarded to the model (used only during
            prototype construction).

        Returns
        -------
        dict[str, jax.Array]
            Mapping from site name to the sampled value in the *constrained*
            space.
        """
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result: dict[str, jax.Array] = {}
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = self._event_dims[name]
            init_loc = self._init_locs[name]
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])

                site_loc = numpyro.param("{}_{}_loc".format(name, self.prefix), init_loc, event_dim=event_dim)
                site_scale = numpyro.param(
                    "{}_{}_scale".format(name, self.prefix),
                    self._get_site_init_scale(name, init_loc),
                    constraint=self.scale_constraint,
                    event_dim=event_dim,
                )

                site_fn = npdist.Normal(site_loc, site_scale).to_event(event_dim)
                if site["fn"].support is constraints.real or (
                    isinstance(site["fn"].support, constraints.independent)
                    and site["fn"].support.base_constraint is constraints.real
                ):
                    result[name] = numpyro.sample(name, site_fn)
                else:
                    with helpful_support_errors(site):
                        transform = biject_to(site["fn"].support)
                    guide_dist = npdist.TransformedDistribution(site_fn, transform)
                    result[name] = numpyro.sample(name, guide_dist)

        return result


def _wrap_guide_with_kl_scale(guide: Callable) -> Callable:
    """Wrap a guide so the per-step ``kl_annealing_factor`` scales its sample sites.

    The returned closure registers a ``numpyro.handlers.scale`` context around the guide
    so that ``log q(z)`` is scaled by the same factor the model applies to ``log p(z)``.
    Without this, ``handlers.scale`` on the model alone would scale ``log p(z)`` but leave
    ``log q(z)`` untouched, so the per-site KL contribution (``log p - log q``) would not
    scale uniformly. The wrapper extracts the factor from the SVI call signature (the final
    positional argument, or the ``kl_annealing_factor`` keyword), defaulting to ``1.0``.

    Exposed at module level so tests can exercise the exact production wrapper.

    Parameters
    ----------
    guide : Callable
        The guide callable to wrap (e.g. an ``AutoNormal`` / ``ParameterizedScaleAutoNormal``
        instance). It is preserved unmodified for downstream ``.median(...)`` extraction.

    Returns
    -------
    Callable
        A ``scaled_guide(*args, **kwargs)`` closure suitable for passing to ``SVI(...)``.
    """

    def scaled_guide(*args: Any, **kwargs: Any):
        # run_svi appends the KL-annealing factor as the final positional argument on every call
        # (svi.init/update: ``*model_args, factor``), regardless of how many model args precede it.
        kl_annealing_factor = args[-1] if args else kwargs.get("kl_annealing_factor", 1.0)
        with numpyro.handlers.scale(scale=kl_annealing_factor):
            return guide(*args, **kwargs)

    return scaled_guide
