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
import inspect
from abc import ABC
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from math import ceil
from types import ModuleType
from typing import Any, ClassVar, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.optim as noptim
import optax
from numpyro.distributions import Bernoulli as NumpyroBernoulli
from numpyro.infer import Trace_ELBO, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_median, init_to_value
from pydantic import (
    ConfigDict,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    conlist,
    field_validator,
    model_validator,
    validate_call,
)

from pybandits.base import (
    BinaryReward,
    PositiveFloat01,
    ProbabilityWeight,
)
from pybandits.model.base import Model, ModelCC, ModelDP, ModelMO
from pybandits.model.bnn._dnn import DNNMixin
from pybandits.model.bnn._guide import ParameterizedScaleAutoNormal
from pybandits.model.bnn._svi import forward_layers, run_svi
from pybandits.model.bnn._typing import (
    ActivationFunctions,
    OptaxKind,
    _Array,
    _numpy_sigmoid,
)
from pybandits.model.bnn.config import (
    BnnLayerParams,
    BnnParams,
    CategoricalFeatureConfig,
    EarlyStopping,
    EmbeddingParams,
    FeaturesConfig,
    VIUpdateKwargs,
)
from pybandits.model.bnn.priors import BaseLocationScaleArray, NormalArray, StudentTArray


class BaseBayesianNeuralNetwork(Model, DNNMixin, ABC):
    """Bayesian Neural Network model for binary classification.

    This class implements a Bayesian Neural Network with an arbitrary number of fully connected layers
    using NumPyro for binary classification tasks. Posterior inference is by Variational Inference (VI).

    References
    ----------
    Bayesian Learning for Neural Networks (Radford M. Neal, 1995)
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=db869fa192a3222ae4f2d766674a378e47013b1b

    Weight Uncertainty in Neural Networks (Blundell, Cornebise, Kavukcuoglu, Wierstra, ICML 2015)
    https://arxiv.org/abs/1505.05424

    Variational Continual Learning (Nguyen, Li, Bui, Turner, ICLR 2018)
    https://arxiv.org/abs/1710.10628

    Parameters
    ----------
    model_params : BnnParams
        The parameters of the Bayesian Neural Network, including weights and biases for each layer and their initial values for resetting
    update_kwargs : VIUpdateKwargs | dict | None, optional
        Keyword arguments for VI training. May be passed as a plain dict (validated and coerced
        at construction) or as the typed :class:`VIUpdateKwargs` model (``num_steps``/``epochs``,
        ``method``, ``optimizer_type``, ``optimizer_kwargs``, ``batch_size``,
        ``early_stopping_kwargs``, ``num_particles``, ``gradient_clip_norm``,
        ``kl_annealing_fraction``, ...). ``None`` uses the defaults.
    activation : str, optional
        The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
    use_residual_connections : bool, optional
        Whether to use residual connections in the network. Residual connections are only added when
        the layer output dimension is greater than or equal to the input dimension (default is False).
    early_stopping_config : EarlyStoppingConfig | None, optional
        Configuration for early stopping during VI training. If None, no early stopping is used (default is None).

    Examples
    --------
    >>> # Create BNN with Student-t priors (default)
    >>> bnn = BayesianNeuralNetwork.cold_start(
    ...     n_features=2,
    ...     hidden_dim_list=[16, 16],
    ...     dist_type="studentt",
    ...     dist_params_init={"mu": 0, "sigma": 1, "nu": 5}
    ... )
    >>> # Create BNN with Normal priors
    >>> bnn = BayesianNeuralNetwork.cold_start(
    ...     n_features=2,
    ...     hidden_dim_list=[16, 16],
    ...     dist_type="normal",
    ...     dist_params_init={"mu": 0, "sigma": 1}
    ... )

    Notes
    -----
    - The model uses the specified activation function for hidden layers and sigmoid activation for the output layer.
    - The output layer is designed for binary classification tasks, with probabilities modeled
      using a Bernoulli likelihood.
    - When use_residual_connections is True, residual connections are added to hidden layers where the output
      dimension is >= input dimension. For expanding dimensions, the residual is zero-padded.
    """

    model_params: BnnParams

    _logit_var_name: ClassVar[str] = "logit"
    _prob_var_name: ClassVar[str] = "prob"
    weight_var_name: ClassVar[str] = "weight"
    bias_var_name: ClassVar[str] = "bias"
    _embedding_var_name: ClassVar[str] = "embedding"
    _distribution_mapping: ClassVar[dict[str, type]] = {"normal": NormalArray, "studentt": StudentTArray}
    _numerical_eps: ClassVar[float] = 1e-6
    _optax_return_types: ClassVar[dict] = {
        "optimizer": optax.GradientTransformation,
        "lr_scheduler": optax.Schedule,
    }
    _optax_required_kwargs: ClassVar[dict] = {
        "optimizer": "learning_rate",
        "lr_scheduler": "init_value",
    }

    def _resolve_optax_fn(self, name: str, kind: OptaxKind) -> Any:
        """Look up an optax function by name using getattr and validate its return type annotation
        and required keyword argument.

        Parameters
        ----------
        name : str
            Name of the optax attribute (e.g. ``"adam"``, ``"exponential_decay"``).
        kind : OptaxKind
            Key into ``_optax_return_types`` (``"optimizer"`` or ``"lr_scheduler"``).

        Returns
        -------
        Any
            The callable found on the ``optax`` module.

        Raises
        ------
        ValueError
            If ``name`` is not a callable attribute of ``optax``, its return-type
            annotation does not match the expected type for ``kind``, or its signature
            does not accept the required keyword argument for ``kind``.
        """
        fn = getattr(optax, name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Invalid {kind}: '{name}' is not a callable attribute of optax.")
        expected = self._optax_return_types[kind]
        sig = inspect.signature(fn)
        return_annotation = sig.return_annotation
        if isinstance(expected, type) and not hasattr(expected, "__origin__"):
            # e.g. GradientTransformation (plain class) — check via issubclass
            valid = isinstance(return_annotation, type) and issubclass(return_annotation, expected)
        else:
            # e.g. optax.Schedule (parameterized generic) — check for equality
            valid = return_annotation == expected
        if not valid:
            raise ValueError(
                f"Invalid {kind}: '{name}' does not return {expected} (got return annotation: {return_annotation})."
            )
        required_kwarg = self._optax_required_kwargs[kind]
        params = sig.parameters
        has_required = required_kwarg in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_required:
            raise ValueError(
                f"Invalid {kind}: '{name}' does not accept the required keyword argument '{required_kwarg}'."
            )
        return fn

    update_kwargs: VIUpdateKwargs | None = None
    activation: ActivationFunctions = "tanh"
    use_residual_connections: bool = False
    feature_config: FeaturesConfig
    random_seed: NonNegativeInt | None = None
    calibrate_output_bias: bool = False
    bias_calibrated: bool = False

    _rng_key: Any = PrivateAttr(default=None)

    _vi_method_config: ClassVar[dict] = {
        "advi": {
            "guide": ParameterizedScaleAutoNormal,
            "loss": TraceMeanField_ELBO,
        },  # assumes weights are independent; loss is the ELBO
        "fullrank_advi": {
            "guide": AutoMultivariateNormal,
            "loss": Trace_ELBO,
        },  # assumes weights are dependent, for full rank covariance matrix
    }

    _approx_history: np.ndarray = PrivateAttr(None)
    _numpy_activation_fn: Callable = PrivateAttr(None)
    _jax_activation_fn: Callable = PrivateAttr(None)
    _obj_optimizer: Any | None = PrivateAttr(None)
    _early_stopping_callback: EarlyStopping | None = PrivateAttr(None)
    _update_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _transfer_learned_keys: ClassVar[tuple[str, ...]] = ("model_params",)
    _transfer_extendable_keys: ClassVar[tuple[str, ...]] = ()
    _transfer_structural_keys: ClassVar[tuple[str, ...]] = ("activation", "use_residual_connections")

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v):
        if v not in cls._jax_activations.keys():
            raise ValueError(
                f"Invalid activation function: {v}. Supported activations are: {list(cls._jax_activations.keys())}"
            )
        return v

    @model_validator(mode="after")
    def validate_bias_calibrated(self) -> "BaseBayesianNeuralNetwork":
        if self.bias_calibrated and not self.calibrate_output_bias:
            raise ValueError("bias_calibrated=True requires calibrate_output_bias=True.")
        return self

    @classmethod
    def get_embedding_var_name(cls, feat_index: int) -> str:
        """Return the NumPyro variable name for a categorical embedding matrix."""
        return f"{cls._embedding_var_name}_{feat_index}"

    @property
    def approx_history(self) -> np.ndarray | None:
        return self._approx_history

    @property
    def obj_optimizer(self) -> Any:
        """The optax optimizer built from ``update_kwargs``, wrapped via ``optax_to_numpyro``."""
        return self._obj_optimizer

    def _prepare_context_arrays(self, context: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """
        Split a numpy context array into numerical and categorical index arrays.

        Numerical columns are all columns not assigned to a categorical feature.
        Categorical columns are extracted by their explicit ``column_index``.

        Parameters
        ----------
        context : np.ndarray of shape (n_samples, input_dim)
            Input array.

        Returns
        -------
        numerical_arr : np.ndarray of shape (n_samples, n_numerical)
            Numerical features, same dtype as input. Empty ``(n_samples, 0)`` if none.
        cat_indices_dict : dict[int, np.ndarray]
            Mapping from feature index (0-based) to int32 array of shape ``(n_samples,)``
            containing integer category codes.
        """
        n_samples = len(context)
        numeric_indices = self.feature_config.numerical_indices

        if numeric_indices:
            numerical_arr = context[:, numeric_indices]
        else:
            numerical_arr = np.empty((n_samples, 0), dtype=context.dtype)

        configs = self.feature_config.categorical_features_configs
        cat_indices_dict = {i: context[:, cfg.column_index].astype(np.int32) for i, cfg in enumerate(configs)}

        return numerical_arr, cat_indices_dict

    def build_optax_optimizer(self, step_size: float | None = None) -> optax.GradientTransformation:
        """Build the raw optax optimizer from ``update_kwargs`` (learning-rate schedule included).

        Gradient clipping and the ``optax_to_numpyro`` wrapping are *not* applied here, so callers
        training several parameter groups at once (see ``CmabMetaModel._get_joint_optimizer``) can
        combine per-group optimizers before clipping globally.

        Parameters
        ----------
        step_size : float | None, default=None
            Learning rate override; ``None`` uses ``optimizer_kwargs["step_size"]`` (default ``0.01``).
        """
        optimizer_kwargs = dict(self.update_kwargs.optimizer_kwargs or {})
        default_step_size = optimizer_kwargs.pop("step_size", 0.01)
        learning_rate = default_step_size if step_size is None else step_size

        # Resolve learning rate (possibly a schedule)
        if self.update_kwargs.lr_scheduler_type is not None:
            scheduler_fn = self._resolve_optax_fn(self.update_kwargs.lr_scheduler_type, "lr_scheduler")
            lr_scheduler_kwargs = self.update_kwargs.lr_scheduler_kwargs or {}
            try:
                learning_rate = scheduler_fn(init_value=learning_rate, **lr_scheduler_kwargs)
            except (TypeError, ValueError) as e:
                raise e.__class__(f"Invalid lr_scheduler_kwargs: {lr_scheduler_kwargs}.\n{e}") from e

        optimizer_fn = self._resolve_optax_fn(self.update_kwargs.optimizer_type, "optimizer")
        try:
            return optimizer_fn(learning_rate=learning_rate, **optimizer_kwargs)
        except (TypeError, ValueError, KeyError) as e:
            raise e.__class__(f"Invalid optimizer kwargs: {optimizer_kwargs}.\n{e}") from e

    def clip_and_wrap_optimizer(self, optimizer: optax.GradientTransformation) -> Any:
        """Chain ``gradient_clip_norm`` (when set) onto ``optimizer`` and wrap it for numpyro."""
        gradient_clip_norm = self.update_kwargs.gradient_clip_norm
        if gradient_clip_norm is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(gradient_clip_norm), optimizer)
        return noptim.optax_to_numpyro(optimizer)

    def _get_obj_optimizer(self) -> Any:
        """The single-model optimizer: ``build_optax_optimizer`` plus clipping and numpyro wrapping."""
        return self.clip_and_wrap_optimizer(self.build_optax_optimizer())

    def _get_early_stopping_callback(self) -> EarlyStopping | None:
        early_stopping_kwargs = self.update_kwargs.early_stopping_kwargs
        if early_stopping_kwargs is not None:
            try:
                return EarlyStopping(**early_stopping_kwargs)
            except Exception as e:
                raise ValueError(f"Invalid early stopping kwargs: {early_stopping_kwargs}.\n{e}")
        return None

    @classmethod
    def get_layer_params_name(cls, layer_ind: PositiveInt) -> tuple[str, str]:
        """NumPyro site names for a layer's weight/bias."""
        weight_layer_params_name = f"{cls.weight_var_name}_{layer_ind}"
        bias_layer_params_name = f"{cls.bias_var_name}_{layer_ind}"
        return weight_layer_params_name, bias_layer_params_name

    @classmethod
    def create_model_params(
        cls,
        feature_config: FeaturesConfig,
        hidden_dim_list: list[PositiveInt] | None,
        use_layerwise_scaling: bool = False,
        dist_class: type[BaseLocationScaleArray] = StudentTArray,
        bias_std: PositiveFloat | None = None,
        **dist_params_init,
    ) -> BnnParams:
        """
        Creates model parameters for a Bayesian neural network (BNN) model according to dist_params_init.
        This method initializes the distribution's parameters for each layer of a BNN
        using the specified number of features, hidden dimensions, and distribution
        initialization parameters.

        Parameters
        ----------
        feature_config : FeaturesConfig
            Full input layout description. First-layer input dimension is ``feature_config.total_output_dim``.
            ``EmbeddingParams`` are created when ``feature_config.categorical_features_configs`` is non-empty.
        hidden_dim_list : list[PositiveInt] | None
            Number of hidden units per hidden layer. If None, no hidden layers are added.
        use_layerwise_scaling : bool
            Whether to use layerwise scaling in the network (default is False).
        dist_class : type
            The distribution class to use for weights, biases, and embeddings, by default ``StudentTArray``.
        bias_std : PositiveFloat | None
            If provided, overrides ``sigma`` from ``dist_params_init`` for all layers' bias priors.
            Applied to every layer's bias (including the output layer's logit bias), leaving weight priors unchanged.
            Default is None (use ``sigma`` from ``dist_params_init``).
        **dist_params_init : dict, optional
            Additional parameters for initializing the distribution of weights and biases.

        Returns
        -------
        BnnParams
            An instance of BnnParams containing the initialized layer parameters.
        """
        effective_n_features = feature_config.total_output_dim

        if hidden_dim_list is None:
            _dim_list = [effective_n_features]
        else:
            _dim_list = [effective_n_features] + hidden_dim_list

        _dim_list.append(1)

        layer_params_init = []
        for layer_ind in range(len(_dim_list) - 1):
            input_dim = _dim_list[layer_ind]
            output_dim = _dim_list[layer_ind + 1]
            w_param = dist_class.cold_start(
                shape=(input_dim, output_dim), use_layerwise_scaling=use_layerwise_scaling, **dist_params_init
            )
            b_dist_params_init = dist_params_init if bias_std is None else {**dist_params_init, "sigma": bias_std}
            b_param = dist_class.cold_start(shape=output_dim, **b_dist_params_init)
            layer_params_init.append(BnnLayerParams(weight=w_param, bias=b_param))

        if feature_config.categorical_features_configs:
            embedding_params = EmbeddingParams.cold_start(
                feature_config=feature_config,
                dist_class=dist_class,
                **dist_params_init,
            )
        else:
            embedding_params = None

        return BnnParams(
            bnn_layer_params=layer_params_init,
            embedding_params=embedding_params,
        )

    def check_context_matrix(self, context: np.ndarray):
        """
        Validate the context input.

        Context must be an array-like with numeric values and the correct number of columns.
        Categorical columns are validated to contain valid integer indices within their vocab range.

        Parameters
        ----------
        context : np.ndarray
            Matrix of contextual features of shape ``(n_samples, n_cols)``.
        """
        expected_cols = self.feature_config.n_features

        try:
            n_cols_context = context.shape[1]
        except Exception as e:
            raise AttributeError(f"Context must be an ArrayLike with {expected_cols} columns: {e}.")

        if not np.issubdtype(context.dtype, np.number):
            raise ValueError("Context array must contain only numeric values.")

        if n_cols_context != expected_cols:
            raise AttributeError(f"Shape mismatch: context must have {expected_cols} columns.")

        configs = self.feature_config.categorical_features_configs
        if configs:
            col_indices = [c.column_index for c in configs]
            cardinalities = np.array([c.cardinality for c in configs])
            raw_cat_cols = context[:, col_indices]
            if not np.all(np.isfinite(raw_cat_cols) & (raw_cat_cols == np.floor(raw_cat_cols))):
                raise ValueError("Categorical feature columns must contain finite integer-valued indices.")
            cat_cols = raw_cat_cols.astype(int)
            in_range = (cat_cols >= 0) & (cat_cols < cardinalities[np.newaxis, :])
            if not np.all(in_range):
                bad_idx = int(np.argmax(~np.all(in_range, axis=0)))
                cfg = configs[bad_idx]
                raise ValueError(
                    f"Categorical feature at column {cfg.column_index} (index {bad_idx}) has indices out of range "
                    f"[0, {cfg.cardinality})."
                )

    @property
    def input_dim(self) -> PositiveInt:
        """
        Returns the number of raw context columns expected by the model.

        Returns
        -------
        PositiveInt
            Equal to ``feature_config.n_features``: the number of columns the
            context numpy array must have.  For categorical models this differs
            from the post-embedding dimension (``feature_config.total_output_dim``).
        """
        return self.feature_config.n_features

    @property
    def resolved_update_kwargs(self) -> dict[str, Any]:
        """The resolved training config (``update_kwargs.model_dump()``, defaults included).

        The public ``update_kwargs`` field is what the user passed (possibly ``None`` or partial);
        this returns the internal, fully-populated config the engine actually trains with — keys such
        as ``method``, ``num_particles`` and ``num_steps`` are guaranteed present. Exposed so the joint
        cMAB meta-model can read a head's effective settings without reaching into a private attribute.

        Returns
        -------
        dict[str, Any]
            The resolved update kwargs used by the SVI training path.
        """
        return self._update_kwargs

    @property
    def hidden_dim_list(self) -> list[int]:
        """
        Returns the hidden layer dimensions of the model.

        Returns
        -------
        list[int]
            Output dimension of each layer except the final output layer.
            Empty list when no hidden layers are present.
        """
        return [layer.weight.shape[1] for layer in self.model_params.bnn_layer_params[:-1]]

    @model_validator(mode="before")
    @classmethod
    def _coerce_update_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Coerce ``update_kwargs`` into the typed ``VIUpdateKwargs`` model.

        Accepts a raw dict (the public API, and the form stored in serialized state), ``None``
        (defaults), or an already-built ``VIUpdateKwargs``. All value validation and defaulting
        lives in the typed model (method validity, ``kl_annealing_fraction`` range, the
        epochs/num_steps precedence warning, ...).
        """
        if not isinstance(data, dict):
            return data
        raw = data.get("update_kwargs")
        if isinstance(raw, VIUpdateKwargs):
            return data
        raw = dict(raw) if raw else {}
        # Copy before mutating: during Union resolution pydantic feeds the same input dict to each
        # candidate member, so mutating it here would leak an injected update_kwargs into sibling
        # models (e.g. QuantitativeBayesianNeuralNetwork, which has no such field).
        data = {**data, "update_kwargs": VIUpdateKwargs(**raw)}
        return data

    def _init_private_attrs(self) -> None:
        """Initialize private attributes that are derived from public fields.

        These attributes (activation functions, optimizer, etc.) are excluded from pickling
        and reconstructed on deserialization.
        """
        self._numpy_activation_fn = self._numpy_activations[self.activation]
        self._jax_activation_fn = self._jax_activations[self.activation]
        self._rng_key = jax.random.PRNGKey(
            self.random_seed
            if self.random_seed is not None
            else int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        )
        self._obj_optimizer = self._get_obj_optimizer()
        self._early_stopping_callback = self._get_early_stopping_callback()

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize activation function PrivateAttr based on the activation setting.
        """
        # Flatten the validated kwargs model into a plain dict for the internal training code,
        # which consumes the merged-with-defaults parameters by key (e.g. ``self._update_kwargs["method"]``).
        self._update_kwargs = self.update_kwargs.model_dump()
        self._init_private_attrs()

    def __getstate__(self) -> dict:
        """Exclude unpicklable private attributes (JAX functions, optimizer objects)."""
        state = self.__dict__.copy()
        # Remove private attrs that hold unpicklable JAX/numpyro objects
        for key in ("_numpy_activation_fn", "_jax_activation_fn", "_obj_optimizer", "_early_stopping_callback"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and reconstruct derived private attributes."""
        self.__dict__.update(state)
        self._init_private_attrs()

    def _forward_layers(
        self,
        next_layer_input: _Array,
        weights_biases: list[tuple[_Array, _Array]],
        activation_fn: Callable[[_Array], _Array],
        linear_fn: Callable[[_Array, _Array, _Array], _Array],
        backend: ModuleType,
    ) -> _Array:
        """
        Shared layer-by-layer forward computation for both JAX/NumPyro and NumPy backends.

        Parameters
        ----------
        next_layer_input : _Array
            Network input, shape ``(batch, input_dim)``. May be a JAX or NumPy array.
        weights_biases : list[tuple[_Array, _Array]]
            Per-layer ``(weights, biases)``. Shapes depend on the backend:
            NumPyro — ``(input_dim, output_dim)`` / ``(output_dim,)``;
            NumPy — ``(n_samples, input_dim, output_dim)`` / ``(n_samples, output_dim)``.
        activation_fn : Callable[[_Array], _Array]
            Activation function matching the backend (``_jax_activation_fn`` or ``_numpy_activation_fn``).
        linear_fn : Callable[[_Array, _Array, _Array], _Array]
            Backend-specific linear transform: ``(x, w, b) -> x @ w + b``.
            For JAX: ``lambda x, w, b: jnp.dot(x, w) + b``.
            For NumPy: ``lambda x, w, b: np.einsum("...i,...ij->...j", x, w) + b``.
        backend : ModuleType
            Array namespace — either ``jnp`` (JAX) or ``np`` (NumPy). Used for
            ``backend.zeros`` and ``backend.concatenate`` in residual padding.

        Returns
        -------
        _Array
            The raw linear output of the final layer (pre-sigmoid).
        """
        return forward_layers(
            next_layer_input=next_layer_input,
            weights_biases=weights_biases,
            activation_fn=activation_fn,
            linear_fn=linear_fn,
            backend=backend,
            use_residual_connections=self.use_residual_connections,
        )

    def _create_update_model(self) -> Callable:
        """
        Create a NumPyro model function for Bayesian Neural Network.

        Reads ``self.model_params.bnn_layer_params`` (the posteriors from the previous update
        round) and feeds them to ``numpyro.sample`` as the priors for this round. The model
        function is consumed by the VI training loop (via ``svi.update``).

        Data is passed as arguments to the returned model function. Minibatching is handled via
        ``numpyro.plate`` with ``subsample_size``. Numerical columns are passed through as-is;
        categorical columns (identified by their ``column_index`` in ``feature_config``) are
        modeled with Bayesian embedding matrices sampled as NumPyro random variables.

        Returns
        -------
        Callable
            NumPyro model function ``model(x, y, kl_annealing_factor=1.0)``. The
            ``kl_annealing_factor`` argument scales the prior (and embedding-prior) ``sample``
            sites' log-probabilities; the likelihood ``out`` site sits outside the scale
            context and is unaffected. The default ``1.0`` makes the scale a numerical no-op.

        Notes
        -----
        The model structure follows these steps:

        1. For each layer, create weight and bias variables from prior distributions
           (current posteriors used as new priors).
        2. Sample embedding matrices for categorical features (if any).
        3. Apply linear transformations and activations through the layers.
        4. Apply sigmoid activation at the output.
        5. Use Bernoulli likelihood for binary classification

        Steps 1-2 happen inside ``numpyro.handlers.scale(scale=kl_annealing_factor)`` so that
        the KL portion of the ELBO can be scheduled across training steps. Step 5 stays outside
        that context so the likelihood term is not scaled.
        """

        batch_size = self._update_kwargs.get("batch_size")

        def model(x: jax.Array, y: jax.Array, kl_annealing_factor: PositiveFloat01 | jax.Array = 1.0):
            self.emit_submodel(x=x, y=y, kl_annealing_factor=kl_annealing_factor, batch_size=batch_size)

        return model

    def emit_submodel(
        self,
        x: jax.Array,
        y: jax.Array,
        kl_annealing_factor: PositiveFloat01 | jax.Array = 1.0,
        batch_size: PositiveInt | None = None,
    ) -> None:
        """Emit this BNN's NumPyro sites on the supplied input array.

        Samples the layer weights/biases and categorical embeddings, runs the forward pass on ``x``,
        and registers the ``logit`` deterministic + ``out`` Bernoulli likelihood. Factored out of
        :meth:`_create_update_model` so a meta-model can compose several arms' sub-models into one
        joint model on a shared backbone embedding *or* on raw context — the meta-model wraps each
        call in ``numpyro.handlers.scope(prefix=arm)`` to keep the (otherwise identical) site names
        from colliding, so this method itself stays namespace-free.

        Parameters
        ----------
        x : jax.Array
            Network input — raw context (standalone / no-backbone head) or a backbone embedding.
        y : jax.Array
            Binary rewards for these rows.
        kl_annealing_factor : PositiveFloat01 | jax.Array
            Scales the prior-site log-probabilities (KL term); ``1.0`` is a no-op.
        batch_size : PositiveInt | None
            Minibatch size for the data plate; ``None`` (or ≥ n_samples) means full batch.
        """
        weights_biases, embedding_matrices = self.sample_head_sites(kl_annealing_factor)
        self._observe(x, y, weights_biases, embedding_matrices, batch_size)

    def sample_head_sites(
        self, kl_annealing_factor: PositiveFloat01 | jax.Array = 1.0
    ) -> tuple[list[tuple[jax.Array, jax.Array]], list[jax.Array]]:
        """Sample this BNN's global latent sites (layer weights/biases + categorical embeddings).

        These are the parameters shared across all rows, so they live *outside* the data plate. The
        ``numpyro.handlers.scale`` wrap lets the per-step KL-annealing factor scale the prior-site
        log-probabilities (combined with the symmetric guide wrap, the full per-site KL is scaled).
        Factored out of :meth:`emit_submodel` so a joint meta-model can sample each arm's head sites,
        stack them across arms, and drive a single minibatched observation.

        Parameters
        ----------
        kl_annealing_factor : PositiveFloat01 | jax.Array
            Scales the prior-site log-probabilities (KL term); ``1.0`` is a no-op.

        Returns
        -------
        tuple[list[tuple[jax.Array, jax.Array]], list[jax.Array]]
            ``(weights_biases, embedding_matrices)`` — per-layer ``(weight, bias)`` and one embedding
            matrix per categorical feature (empty when the head has no categorical embeddings).
        """
        cat_configs = self.feature_config.categorical_features_configs
        has_embeddings = self.model_params.embedding_params is not None and len(cat_configs) > 0
        weights_biases: list[tuple[jax.Array, jax.Array]] = []
        embedding_matrices: list[jax.Array] = []
        with numpyro.handlers.scale(scale=kl_annealing_factor):
            for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
                weight_name, bias_name = self.get_layer_params_name(layer_ind)
                w = numpyro.sample(weight_name, layer_params.weight.to_numpyro_distribution())
                b = numpyro.sample(bias_name, layer_params.bias.to_numpyro_distribution())
                weights_biases.append((w, b))

            if has_embeddings:
                for i, emb_dist in enumerate(self.model_params.embedding_params.embeddings):
                    emb = numpyro.sample(self.get_embedding_var_name(i), emb_dist.to_numpyro_distribution())
                    embedding_matrices.append(emb)
        return weights_biases, embedding_matrices

    def _observe(
        self,
        x: jax.Array,
        y: jax.Array,
        weights_biases: list[tuple[jax.Array, jax.Array]],
        embedding_matrices: list[jax.Array],
        batch_size: PositiveInt | None = None,
    ) -> None:
        """Forward the (optionally minibatched) input through the sampled sites and observe ``out``.

        Builds the network input (numerical columns + embedded categoricals), runs the forward pass,
        registers the ``logit`` deterministic, and observes the ``out`` Bernoulli likelihood — inside a
        subsampling ``data`` plate when ``batch_size`` is set. Kept separate from
        :meth:`sample_head_sites` so the latent sites can be sampled once and reused.

        Parameters
        ----------
        x : jax.Array
            Network input — raw context (standalone / no-backbone head) or a backbone embedding.
        y : jax.Array
            Binary rewards for these rows.
        weights_biases : list[tuple[jax.Array, jax.Array]]
            Per-layer ``(weight, bias)`` from :meth:`sample_head_sites`.
        embedding_matrices : list[jax.Array]
            Categorical embedding matrices from :meth:`sample_head_sites` (empty if none).
        batch_size : PositiveInt | None
            Minibatch size for the data plate; ``None`` (or ≥ n_samples) means full batch.
        """
        numerical_indices = self.feature_config.numerical_indices
        cat_configs = self.feature_config.categorical_features_configs
        has_embeddings = self.model_params.embedding_params is not None and len(cat_configs) > 0
        n_samples = x.shape[0]

        # Data plate with optional minibatching
        if batch_size is not None and batch_size < n_samples:
            plate_ctx = numpyro.plate("data", size=n_samples, subsample_size=batch_size)
        else:
            plate_ctx = nullcontext()

        with plate_ctx as idx:
            x_batch = x[idx] if idx is not None else x
            y_batch = y[idx] if idx is not None else y

            # Build network input: numerical features + embedded categoricals
            if has_embeddings:
                input_parts = []
                if numerical_indices:
                    input_parts.append(x_batch[:, numerical_indices])
                for i, cfg in enumerate(cat_configs):
                    cat_idx = x_batch[:, cfg.column_index].astype(jnp.int32)
                    input_parts.append(embedding_matrices[i][cat_idx])
                next_layer_input = jnp.concatenate(input_parts, axis=1) if len(input_parts) > 1 else input_parts[0]
            else:
                next_layer_input = x_batch

            # Forward pass
            linear_transform = self._forward_layers(
                next_layer_input=next_layer_input,
                weights_biases=weights_biases,
                activation_fn=self._jax_activation_fn,
                linear_fn=lambda x, w, b: jnp.dot(x, w) + b,
                backend=jnp,
            )

            # Final output processing
            logit = numpyro.deterministic(
                self._logit_var_name,
                linear_transform.squeeze(-1),
            )
            numpyro.sample(
                "out", NumpyroBernoulli(logits=logit), obs=y_batch
            )  # "The observed reward follows a Bernoulli distribution given the network output"

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_weights(self, n_samples: PositiveInt, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Sample weights and biases for each sample and each layer.

        Parameters
        ----------
        n_samples : PositiveInt
            The number of samples (users) to draw weights for. Must be positive.
        rng : np.random.Generator
            Numpy random generator forwarded to numpy samplers. Enables reproducible
            weight sampling and integration with the central MAB-level RNG.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            A list of length num_layers, where each element is (weights, biases) for that layer.
            - weights shape: (n_samples, input_dim, output_dim)
            - biases shape: (n_samples, output_dim)
        """
        sampled_weights = []
        for layer_params in self.model_params.bnn_layer_params:
            input_dim = layer_params.weight.shape[0]
            output_dim = layer_params.weight.shape[1]

            w = layer_params.weight.sample_rvs(size=(n_samples, input_dim, output_dim), rng=rng)
            b = layer_params.bias.sample_rvs(size=(n_samples, output_dim), rng=rng)

            sampled_weights.append((w, b))

        return sampled_weights

    def sample_embeddings(self, context: np.ndarray, rng: np.random.Generator) -> list[np.ndarray] | None:
        """
        Sample embedding vectors for each categorical feature given the context.

        For each categorical feature, extracts the integer indices from the context
        and samples from the corresponding embedding distribution at those indices.

        Parameters
        ----------
        context : np.ndarray
            Context matrix, shape ``(n_samples, feature_config.n_features)``.
            Categorical columns contain integer indices into the embedding vocabulary.
        rng : np.random.Generator
            Numpy random generator forwarded to numpy samplers.

        Returns
        -------
        list[np.ndarray] | None
            One array per categorical feature, each of shape ``(n_samples, emb_dim)``.
            ``None`` when the model has no categorical features.
        """
        if not self.feature_config.has_categorical:
            return None
        _context = np.atleast_2d(context)
        _, cat_indices_dict = self._prepare_context_arrays(_context)
        return [
            self.model_params.embedding_params.embeddings[i]
            .sample_at_indices(cat_indices_dict[i], rng=rng)
            .reshape(len(_context), -1)
            for i in range(len(self.feature_config.categorical_features_configs))
        ]

    @staticmethod
    def extract_sample(
        sampled_weights: list[tuple[np.ndarray, np.ndarray]],
        sampled_embeddings: list[np.ndarray] | None,
        sample_idx: NonNegativeInt,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray] | None]:
        """
        Extract the weights, biases, and embeddings for a specific sample.

        Parameters
        ----------
        sampled_weights : list[tuple[np.ndarray, np.ndarray]]
            List of (weights, biases) per layer.
            Each weights has shape (n_samples, input_dim, output_dim), biases (n_samples, output_dim).
        sampled_embeddings : list[np.ndarray] | None
            Pre-sampled embedding vectors, one per categorical feature, each of shape
            ``(n_samples, emb_dim)``. ``None`` when no categorical features.
        sample_idx : NonNegativeInt
            The index of the sample to extract.

        Returns
        -------
        tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray] | None]
            ``(weights_idx, embeddings_idx)`` sliced to a single sample (batch dim = 1).
        """
        weights_idx = [(w[sample_idx : sample_idx + 1], b[sample_idx : sample_idx + 1]) for w, b in sampled_weights]
        embeddings_idx = (
            [ev[sample_idx : sample_idx + 1] for ev in sampled_embeddings] if sampled_embeddings is not None else None
        )
        return weights_idx, embeddings_idx

    def _prepare_forward_input(self, context: np.ndarray, sampled_embeddings: list[np.ndarray] | None) -> np.ndarray:
        """
        Replace categorical columns with pre-sampled embedding vectors.

        Parameters
        ----------
        context : np.ndarray
            Context matrix, shape ``(n_samples, feature_config.n_features)``.
        sampled_embeddings : list[np.ndarray] | None
            One array per categorical feature, each of shape ``(n_samples, emb_dim)``.
            ``None`` when the model has no categorical features.

        Returns
        -------
        np.ndarray
            Network input with categorical columns replaced by embedding vectors,
            shape ``(n_samples, feature_config.total_output_dim)``.
        """
        if sampled_embeddings is None:
            return context
        _context = np.atleast_2d(context)
        numerical_arr, _ = self._prepare_context_arrays(_context)
        parts = [numerical_arr] if numerical_arr.shape[1] > 0 else []
        parts.extend(sampled_embeddings)
        return np.concatenate(parts, axis=1)

    def forward_pass(
        self,
        sampled_weights: list[tuple[np.ndarray, np.ndarray]],
        context: np.ndarray,
        sampled_embeddings: list[np.ndarray] | None = None,
    ) -> list[ProbabilityWeight]:
        """
        Apply the neural network forward pass using pre-sampled weights, biases, and embeddings.

        All stochastic parameters must be sampled externally (via ``sample_weights`` and
        ``sample_embeddings``) before calling this method.

        Parameters
        ----------
        sampled_weights : list[tuple[np.ndarray, np.ndarray]]
            List of (weights, biases) per layer from ``sample_weights``.
            Each weights has shape (n_samples, input_dim, output_dim), biases (n_samples, output_dim).
        context : np.ndarray
            Context matrix, shape (n_samples, feature_config.n_features).
            Categorical columns contain integer indices into the embedding vocabulary.
        sampled_embeddings : list[np.ndarray] | None
            Pre-sampled embedding vectors from ``sample_embeddings``, one array per
            categorical feature, each of shape ``(n_samples, emb_dim)``.
            ``None`` when the model has no categorical features.

        Returns
        -------
        list[ProbabilityWeight]
            Each element is (probability, weighted_sum) per sample.
        """
        next_layer_input = self._prepare_forward_input(context, sampled_embeddings)

        linear_transform = self._forward_layers(
            next_layer_input=next_layer_input,
            weights_biases=sampled_weights,
            activation_fn=self._numpy_activation_fn,
            linear_fn=lambda x, w, b: np.einsum("...i,...ij->...j", x, w) + b,
            backend=np,
        )

        weighted_sum = linear_transform.squeeze(-1)
        prob = _numpy_sigmoid(weighted_sum)
        return list(zip(prob, weighted_sum))

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_proba(self, context: np.ndarray, rng: np.random.Generator) -> list[ProbabilityWeight]:
        """
        Samples probabilities and logits from the prior predictive distribution.

        Parameters
        ----------
        context : np.ndarray
            The context matrix for which the probabilities are to be sampled.
        rng : np.random.Generator
            Numpy random generator for weight/embedding sampling.
            The JAX ``_rng_key`` remains the authority for VI training.

        Returns
        -------
        list[ProbabilityWeight]
            Each element is a tuple containing the probability of a positive reward and
            the network logit.
        """
        self.check_context_matrix(context=context)

        _context = np.atleast_2d(context)
        n_samples = len(_context)

        sampled_weights = self.sample_weights(n_samples, rng=rng)
        sampled_embeddings = self.sample_embeddings(_context, rng=rng)

        return self.forward_pass(
            sampled_weights=sampled_weights, context=_context, sampled_embeddings=sampled_embeddings
        )

    def _create_updated_layer_params(
        self,
        w_mu: np.ndarray,
        w_sigma: np.ndarray,
        b_mu: np.ndarray,
        b_sigma: np.ndarray,
        original_weight: BaseLocationScaleArray,
        original_bias: BaseLocationScaleArray,
    ) -> BnnLayerParams:
        """
        Create updated layer parameters with new mu and sigma values, preserving distribution type.

        Parameters
        ----------
        w_mu : np.ndarray
            Updated weight mean values.
        w_sigma : np.ndarray
            Updated weight standard deviation values.
        b_mu : np.ndarray
            Updated bias mean values.
        b_sigma : np.ndarray
            Updated bias standard deviation values.
        original_weight : BaseLocationScaleArray
            Original weight distribution to preserve type and nu (if StudentTArray).
        original_bias : BaseLocationScaleArray
            Original bias distribution to preserve type and nu (if StudentTArray).

        Returns
        -------
        BnnLayerParams
            New layer parameters with updated values.
        """

        updated_weight = original_weight.with_dist_parameters(mu=w_mu.tolist(), sigma=w_sigma.tolist())
        updated_bias = original_bias.with_dist_parameters(mu=b_mu.tolist(), sigma=b_sigma.tolist())

        return BnnLayerParams(weight=updated_weight, bias=updated_bias)

    def update_embedding_params_from_vi(self, site_mu: dict, site_sigma: dict) -> None:
        """
        Update embedding matrices from per-site VI posterior means and stds.

        Parameters
        ----------
        site_mu : dict
            Per-site mean arrays keyed by variable name.
        site_sigma : dict
            Per-site std arrays keyed by variable name.
        """
        if self.model_params.embedding_params is None:
            return
        updated_embeddings = []
        for i, orig_emb in enumerate(self.model_params.embedding_params.embeddings):
            emb_var_name = self.get_embedding_var_name(i)
            emb_shape = orig_emb.shape
            emb_mu = np.array(site_mu[emb_var_name]).reshape(emb_shape)
            emb_sigma = np.array(site_sigma[emb_var_name]).reshape(emb_shape)
            updated_embeddings.append(orig_emb.with_dist_parameters(mu=emb_mu.tolist(), sigma=emb_sigma.tolist()))
        self.model_params.embedding_params.embeddings = updated_embeddings

    def _build_svi_guide_init(self) -> tuple:
        """Build ``init_loc_fn`` and ``init_scale_fn`` for the SVI guide.

        ``init_loc_fn`` is ``init_to_value`` seeded with the current model mu values, unless all
        mu values are zero (cold start), in which case ``init_to_median`` is used instead.
        ``init_scale_fn`` maps each site name to its current sigma array so the
        :class:`ParameterizedScaleAutoNormal` guide starts with the correct per-parameter width.
        Unknown site names (e.g. for fullrank_advi's scalar fallback) return the global avg sigma.

        Returns
        -------
        tuple
            ``(init_loc_fn, init_scale_fn)``
        """
        values, site_sigmas, all_mus, all_sigmas = self.collect_guide_init_arrays()
        return self.build_guide_init_fns(values, site_sigmas, all_mus, all_sigmas)

    def collect_guide_init_arrays(self) -> tuple[dict, dict, list, list]:
        """Collect this BNN's guide-init contributions, keyed by (namespace-free) site name.

        Returns ``(values, site_sigmas, all_mus, all_sigmas)``. A joint multi-arm meta-model collects
        these per arm and re-keys them under each arm's scope prefix before building a single pair of
        init functions via :meth:`build_guide_init_fns`.
        """
        values: dict = {}
        all_mus: list = []
        all_sigmas: list = []
        site_sigmas: dict = {}

        for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
            w_name, b_name = self.get_layer_params_name(layer_ind)
            values[w_name] = jnp.array(layer_params.weight.params["mu"])
            values[b_name] = jnp.array(layer_params.bias.params["mu"])
            all_mus.append(layer_params.weight.params["mu"].ravel())
            all_mus.append(layer_params.bias.params["mu"].ravel())
            all_sigmas.append(layer_params.weight.params["sigma"].ravel())
            all_sigmas.append(layer_params.bias.params["sigma"].ravel())
            site_sigmas[w_name] = layer_params.weight.params["sigma"]
            site_sigmas[b_name] = layer_params.bias.params["sigma"]

        if self.model_params.embedding_params is not None:
            for i, emb in enumerate(self.model_params.embedding_params.embeddings):
                emb_name = self.get_embedding_var_name(i)
                values[emb_name] = jnp.array(emb.params["mu"])
                all_mus.append(emb.params["mu"].ravel())
                all_sigmas.append(emb.params["sigma"].ravel())
                site_sigmas[emb_name] = emb.params["sigma"]

        return values, site_sigmas, all_mus, all_sigmas

    def build_guide_init_fns(self, values: dict, site_sigmas: dict, all_mus: list, all_sigmas: list) -> tuple:
        """Build ``(init_loc_fn, init_scale_fn)`` from collected guide-init arrays (see
        :meth:`collect_guide_init_arrays`). Shared by the standalone BNN and the joint meta-model."""
        all_mus_flat = np.concatenate(all_mus)
        init_loc_fn = init_to_median if np.all(all_mus_flat == 0) else init_to_value(values=values)
        avg_sigma = float(max(np.mean(np.concatenate(all_sigmas)), self._numerical_eps))
        site_sigmas = {name: np.maximum(sigma, self._numerical_eps) for name, sigma in site_sigmas.items()}
        init_scale_fn = lambda name: site_sigmas.get(name, avg_sigma)  # noqa: E731
        return init_loc_fn, init_scale_fn

    def compute_epoch_steps(self, n_samples: int) -> list[int]:
        """Epoch-step partition for the SVI run from this BNN's ``_update_kwargs``.

        Shared by the standalone ``_run_svi_training_loop`` and the joint cMAB engine so that
        ``epochs``/``batch_size``/``num_steps`` settings are honoured identically on both paths.
        """
        effective_batch_size = self._update_kwargs.get("batch_size") or n_samples
        steps_per_epoch = max(1, n_samples // effective_batch_size)
        if self._update_kwargs.get("epochs") is not None:
            return [steps_per_epoch] * self._update_kwargs["epochs"]
        num_steps = self._update_kwargs["num_steps"]
        n_full_epochs, remaining = np.divmod(num_steps, steps_per_epoch)
        epoch_steps_list = [steps_per_epoch] * n_full_epochs
        if remaining > 0:
            epoch_steps_list.append(remaining)
        return epoch_steps_list or [1]

    def build_kl_annealing_factors(self, epoch_steps_list: list[int]) -> list[jnp.ndarray]:
        """Build the per-step KL annealing factor schedule, split into per-epoch chunks.

        The factor ``β(step)`` multiplies the KL portion of the ELBO at each SVI step:

        - Inactive (``kl_annealing_fraction is None``): every factor is ``1.0``, making the
          ``handlers.scale`` wrap a numerical no-op. The same code path runs in both active
          and inactive cases so there is no feature-gate branching in the training loop.
        - Active: linear ramp ``β(step) = min(1, (step + 1) / W)`` where
          ``W = max(1, ceil(kl_annealing_fraction * total_steps))``. The ``+1`` ensures
          ``β(0) = 1/W > 0`` so the KL term is never fully switched off at the first step.
          The effective domain is the half-open interval ``(0, 1]``, not ``[0, 1]``.

        Parameters
        ----------
        epoch_steps_list : list[int]
            Number of SVI steps per epoch. Their sum is the total number of training steps.

        Returns
        -------
        list[jnp.ndarray]
            One 1-D ``jnp.ndarray`` per epoch, fed as the ``xs`` argument of ``jax.lax.scan``
            in the training loop. Each element holds the per-step factor for that epoch.
        """
        kl_annealing_fraction = self._update_kwargs.get("kl_annealing_fraction")
        total_steps = int(sum(epoch_steps_list))

        if kl_annealing_fraction is None:
            factor_array = jnp.ones(total_steps)
        else:
            warmup_steps = max(1, ceil(float(kl_annealing_fraction) * total_steps))
            factor_array = jnp.minimum(1.0, (jnp.arange(total_steps) + 1) / warmup_steps)

        # Split into per-epoch slices using cumulative epoch boundaries.
        split_points = np.cumsum(epoch_steps_list)[:-1].tolist()
        return list(jnp.split(factor_array, split_points))

    def _run_svi_training_loop(self, x_jnp: jnp.ndarray, y_jnp: jnp.ndarray, n_samples: int) -> tuple:
        """
        Set up and run the SVI training loop (the per-update VI optimization), returning ``(svi, guide, params)``.

        Each SVI step:
        (1) samples weights from the GUIDE (current approximate posterior),
        (2) runs the forward pass with those weights,
        (3) computes ELBO =
        ``log_likelihood(data | sampled_weights) - KL(guide || prior)`` — where the KL term
        may be scaled by the per-step ``kl_annealing_factor`` when annealing is active,
        (4) computes gradients of ``-ELBO`` w.r.t. the guide's mu/sigma,
        (5) lets the optimizer update them.
        All of (1)-(5) happen inside one ``svi.update()`` call, with
        the loop body wrapped in ``jax.lax.scan`` for speed.

        Parameters
        ----------
        x_jnp : jnp.ndarray
            Context array in JAX format.
        y_jnp : jnp.ndarray
            Rewards array in JAX format.
        n_samples : int
            Number of data points (used for ELBO scaling and epoch computation).

        Returns
        -------
        tuple
            ``(svi, guide, params)`` where *svi* is the configured :class:`SVI` object
            (built against a scale-wrapped guide closure that is a numerical no-op when
            ``kl_annealing_fraction`` is ``None``), *guide* is the underlying ``AutoGuide``
            instance (returned unwrapped so that downstream ``guide.median(...)`` extraction
            keeps working), and *params* are the final variational parameters.
        """
        if self._early_stopping_callback is not None:
            self._early_stopping_callback.reset()
        _model = self._create_update_model()

        # Build the per-step KL-annealing factor schedule for the entire run
        epoch_factor_arrays = self.build_kl_annealing_factors(self.compute_epoch_steps(n_samples))

        # Set up VI method (guide + loss) dynamically
        vi_method = self._update_kwargs["method"]
        method_config = self._vi_method_config[vi_method]
        init_loc_fn, init_scale_fn = self._build_svi_guide_init()
        if vi_method == "advi":
            # Posterior initialization: mu=median of prior (per-site), sigma=per-site init_scale.
            guide = ParameterizedScaleAutoNormal(
                _model,
                init_loc_fn=init_loc_fn,
                init_scale_fn=init_scale_fn,
            )
        else:
            # fullrank_advi needs a scalar init_scale; use the avg sigma via the unknown-site fallback.
            guide = method_config["guide"](_model, init_loc_fn=init_loc_fn, init_scale=init_scale_fn(""))

        num_particles = self._update_kwargs["num_particles"]
        loss = method_config["loss"](num_particles=num_particles)
        restore_best = self._update_kwargs.get("restore_best_svi_state", True)

        # Build + drive the SVI loop via the shared `run_svi` helper (guide-wrapping, SVI, jit/scan
        # with x/y as explicit jit args, and the epoch loop). `guide` stays unwrapped here for any
        # downstream `.median(...)` extraction. The same helper powers the joint cMAB engine.
        svi, params, self._approx_history, self._rng_key = run_svi(
            model=_model,
            guide=guide,
            optimizer=self._obj_optimizer,
            loss=loss,
            rng_key=self._rng_key,
            model_args=(x_jnp, y_jnp),
            epoch_factor_arrays=epoch_factor_arrays,
            restore_best=restore_best,
            early_stopping_callback=self._early_stopping_callback,
        )
        return svi, guide, params

    def extract_advi_params(self, params: dict) -> tuple:
        """
        Extract per-site posterior means and stds from AutoNormal (ADVI) guide params.

        Parameters
        ----------
        params : dict
            Raw variational parameters returned by ``svi.get_params()``.

        Returns
        -------
        tuple
            ``(site_mu, site_sigma)`` dicts mapping site name → array.
        """
        # AutoNormal stores per-site: {name}_auto_loc (mean) and {name}_auto_scale (already-constrained std)
        site_mu = {k.removesuffix("_auto_loc"): v for k, v in params.items() if k.endswith("_auto_loc")}
        site_sigma = {
            k.removesuffix("_auto_scale"): np.maximum(v, self._numerical_eps)
            for k, v in params.items()
            if k.endswith("_auto_scale")
        }
        return site_mu, site_sigma

    def _extract_fullrank_advi_params(self, guide, params: dict) -> tuple:
        """
        Extract per-site posterior means and stds from AutoMultivariateNormal (full-rank ADVI) guide params.

        Parameters
        ----------
        guide : numpyro AutoGuide
            The fitted full-rank ADVI guide.
        params : dict
            Raw variational parameters returned by ``svi.get_params()``.

        Returns
        -------
        tuple
            ``(site_mu, site_sigma)`` dicts mapping site name → array.
        """
        # AutoMultivariateNormal stores a joint loc vector and scale_tril matrix.
        # Extract exact marginal means and stds directly (no sampling).
        scale_tril = params["auto_scale_tril"]
        marginal_std = jnp.linalg.norm(scale_tril, axis=1)
        # guide.median() may include deterministic sites (e.g. logit) which are
        # not in auto_loc/scale_tril. Filter to only sampled BNN sites so the
        # offset slicing into marginal_std stays in bounds.
        site_mu_all = guide.median(params)
        sampled_site_names: set = set()
        for _layer_ind in range(len(self.model_params.bnn_layer_params)):
            _w_name, _b_name = self.get_layer_params_name(_layer_ind)
            sampled_site_names.add(_w_name)
            sampled_site_names.add(_b_name)
        if self.model_params.embedding_params is not None:
            for _i in range(len(self.model_params.embedding_params.embeddings)):
                sampled_site_names.add(self.get_embedding_var_name(_i))
        site_mu = {k: v for k, v in site_mu_all.items() if k in sampled_site_names}
        offset = 0
        site_sigma = {}
        for name, val in site_mu.items():
            n = val.size
            site_sigma[name] = marginal_std[offset : offset + n].reshape(val.shape)
            offset += n
        return site_mu, site_sigma

    def _extract_vi_params(self, x_jnp: jnp.ndarray, y_jnp: jnp.ndarray, n_samples: int) -> list:
        """
        Run SVI, extract per-site posteriors, and return updated layer params.

        Parameters
        ----------
        x_jnp : jnp.ndarray
            Context array in JAX format.
        y_jnp : jnp.ndarray
            Rewards array in JAX format.
        n_samples : int
            Number of data points.

        Returns
        -------
        List
            Updated ``BnnLayerParams`` list (embeddings are updated in-place as a side-effect).
        """
        svi, guide, params = self._run_svi_training_loop(x_jnp, y_jnp, n_samples)

        vi_method = self._update_kwargs["method"]
        if vi_method == "advi":
            site_mu, site_sigma = self.extract_advi_params(params)
        elif vi_method == "fullrank_advi":
            site_mu, site_sigma = self._extract_fullrank_advi_params(guide, params)
        else:
            raise ValueError(
                f"Invalid VI method: {vi_method}. Supported methods are: {list(self._vi_method_config.keys())}"
            )

        # Update layer params + embeddings from per-site posterior
        updated_layer_params_list = self.layer_params_from_posterior(site_mu, site_sigma)
        self.update_embedding_params_from_vi(site_mu, site_sigma)
        return updated_layer_params_list

    def layer_params_from_posterior(self, site_mu: dict, site_sigma: dict) -> list:
        """Build updated ``BnnLayerParams`` from a per-site posterior dict keyed by (namespace-free) name.

        Reused by the joint multi-arm meta-model: after one SVI pass, the meta-model slices each arm's
        scoped sites back to namespace-free names and calls this to read back that arm's layers.
        """
        updated_layer_params_list = []
        for layer_ind, layer_params in enumerate(self.model_params.bnn_layer_params):
            weight_layer_params_name, bias_layer_params_name = self.get_layer_params_name(layer_ind)
            w_shape = layer_params.weight.shape
            b_shape = layer_params.bias.shape

            w_mu = np.array(site_mu[weight_layer_params_name]).reshape(w_shape)
            w_sigma = np.array(site_sigma[weight_layer_params_name]).reshape(w_shape)
            b_mu = np.array(site_mu[bias_layer_params_name]).reshape(b_shape)
            b_sigma = np.array(site_sigma[bias_layer_params_name]).reshape(b_shape)

            updated_layer_params = self._create_updated_layer_params(
                w_mu, w_sigma, b_mu, b_sigma, layer_params.weight, layer_params.bias
            )
            updated_layer_params_list.append(updated_layer_params)
        return updated_layer_params_list

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _update(self, context: np.ndarray, rewards: list[BinaryReward]):
        """
        Update the model_params with new context and rewards.

        Parameters
        ----------
        context : np.ndarray
            The context matrix where each row represents a context vector.
        rewards : list[BinaryReward]
            A list of binary rewards corresponding to each context vector.

        Notes
        -----
        This method updates the model's parameters by fitting the posterior distribution with
        Variational Inference (VI).
        """
        self.check_context_matrix(context=context)

        if len(context) != len(rewards):
            raise AttributeError("Shape mismatch: context and rewards must have the same length.")

        _context = np.atleast_2d(context)
        x_jnp = jnp.array(_context, dtype=jnp.float32)
        y_jnp = jnp.array(np.array(rewards, dtype=np.int32), dtype=jnp.int32)
        n_samples = _context.shape[0]

        if self.calibrate_output_bias:
            self._calibrate_output_bias(rewards)

        # Forgetting: widen the current posterior (used as the prior for this fit) before re-fitting,
        # so fresh data dominates old evidence. No-op on the first fit (no posterior to inflate yet),
        # and no-op when decay_factor is None or 1.
        is_first_fit = self.n_successes == self._prior_pseudo_count and self.n_failures == self._prior_pseudo_count
        if not is_first_fit:
            self._inflate_prior_variance()

        self.model_params.bnn_layer_params = self._extract_vi_params(x_jnp, y_jnp, n_samples)

    def _inflate_prior_variance(self):
        """
        Inflate the stored posterior variance before re-fitting, implementing per-update forgetting.
        Every weight, bias, and embedding ``sigma`` is multiplied
        by ``1 / decay_factor``, widening the prior consumed by the next VI round so that fresh
        data dominates old evidence. The means (``mu``) and the Student-t degrees of freedom (``nu``)
        are left untouched. No-op when ``decay_factor`` is None or 1.
        """
        if self.decay_factor is None or self.decay_factor == 1:
            return
        inflation = 1.0 / self.decay_factor
        for layer_params in self.model_params.bnn_layer_params:
            layer_params.weight = layer_params.weight.with_dist_parameters(
                sigma=(np.asarray(layer_params.weight.params["sigma"]) * inflation).tolist()
            )
            layer_params.bias = layer_params.bias.with_dist_parameters(
                sigma=(np.asarray(layer_params.bias.params["sigma"]) * inflation).tolist()
            )
        if self.model_params.embedding_params is not None:
            self.model_params.embedding_params.embeddings = [
                emb.with_dist_parameters(sigma=(np.asarray(emb.params["sigma"]) * inflation).tolist())
                for emb in self.model_params.embedding_params.embeddings
            ]

    @classmethod
    @validate_call
    def cold_start(
        cls,
        n_features: PositiveInt,
        hidden_dim_list: list[PositiveInt] | None = None,
        update_kwargs: dict | None = None,
        dist_type: Literal["normal", "studentt"] = "studentt",
        dist_params_init: dict[str, float] | None = None,
        activation: ActivationFunctions = "tanh",
        use_residual_connections: bool = False,
        use_layerwise_scaling: bool = False,
        bias_std: PositiveFloat | None = None,
        categorical_features: dict[NonNegativeInt, NonNegativeInt] | None = None,
        random_seed: NonNegativeInt | None = None,
        calibrate_output_bias: bool = False,
        decay_factor: PositiveFloat01 | None = None,
        **kwargs,
    ) -> Self:
        """
        Initialize a Bayesian Neural Network with a cold start.

        Parameters
        ----------
        n_features : PositiveInt
            Total number of columns in the context array, including any categorical columns.
        hidden_dim_list : list[PositiveInt] | None, optional
            List of dimensions for the hidden layers of the network. If None, no hidden layers are added.
        update_kwargs : dict | None, optional
            Additional keyword arguments for VI training. Default is None.
        dist_type : Literal["normal", "studentt"]
            Type of distribution to use for priors. Default is "studentt".
        dist_params_init : dict[str, float] | None, optional
            Initial distribution parameters for the network weights and biases. Default is None.
            For Student-t distributions: requires "mu", "sigma", and "nu" parameters.
            For Normal distributions: requires "mu" and "sigma" parameters (no "nu" needed).
        activation : str
            The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
        use_residual_connections : bool
            Whether to use residual connections in the network (default is False).
        use_layerwise_scaling : bool
            Whether to use layerwise scaling in the network (default is False).
            When applied, the sigma is scaled by the square root of the input dimension.
            This is useful to enable smoother convergence with Gaussian Process-like behavior.
        bias_std : PositiveFloat | None
            If provided, overrides ``sigma`` from ``dist_params_init`` for all layers' bias priors,
            leaving weight priors untouched. Useful to restrain the prior on the output-layer logit
            (which otherwise pushes mass towards p=0 / p=1 after sigmoid at cold start). Default is None.
        calibrate_output_bias : bool
            If True, the output-layer bias mu is set to ``logit(empirical_reward_rate)`` on the very first
            ``update()`` call, before VI training begins. This replaces the cold-start prior mean
            (logit 0 ≈ 50 % reward rate) with a data-driven intercept, preventing optimistic over-exploration
            of arms that have accumulated little data. The calibration fires only once per arm lifetime
            (or after a ``_reset()``). Default is False.
        categorical_features : dict[int, int] | None, optional
            Categorical columns as ``{column_index: cardinality}``. Each categorical column is
            modelled with a Bayesian embedding matrix; ``embedding_dim`` is set automatically
            to ``ceil(cardinality / embedding_dim_divisor)``. Columns absent from this dict are treated as numerical.
        random_seed : NonNegativeInt | None, optional
            Seed for the JAX PRNG key. If None, a seed is drawn from OS entropy at construction time
            and stored on the instance, so the same initial key is reproduced after serialization.
            Pass an explicit integer for fully reproducible runs.
        decay_factor : PositiveFloat01 | None
            Per-update forgetting factor in (0, 1]. When set, the weight/bias/embedding posterior
            variances are inflated by ``1 / decay_factor`` before each re-fit. Default is None.
        **kwargs
            Additional keyword arguments for the BayesianNeuralNetwork constructor.

        Returns
        -------
        Self
            An instance of the Bayesian Neural Network initialized with the specified parameters.
        """
        cat_configs = [
            CategoricalFeatureConfig(
                column_index=col_idx,
                cardinality=cardinality,
                embedding_dim=ceil(cardinality / cls.embedding_dim_divisor),
            )
            for col_idx, cardinality in (categorical_features or {}).items()
        ]
        feature_config = FeaturesConfig(n_features=n_features, categorical_features_configs=cat_configs)

        dist_params_init = (dist_params_init or {}).copy()

        if dist_type not in cls._distribution_mapping:
            raise ValueError(
                f"Invalid dist_type: {dist_type}. Must be one of {list(cls._distribution_mapping.keys())}. "
                f"Example: dist_type='normal', dist_params_init={{'mu': 0, 'sigma': 1}}"
            )

        dist_class = cls._distribution_mapping[dist_type]
        model_params = cls.create_model_params(
            feature_config=feature_config,
            hidden_dim_list=hidden_dim_list,
            use_layerwise_scaling=use_layerwise_scaling,
            dist_class=dist_class,
            bias_std=bias_std,
            **dist_params_init,
        )
        return cls(
            model_params=model_params,
            update_kwargs=update_kwargs,
            activation=activation,
            use_residual_connections=use_residual_connections,
            feature_config=feature_config,
            random_seed=random_seed,
            calibrate_output_bias=calibrate_output_bias,
            decay_factor=decay_factor,
            **kwargs,
        )

    def _calibrate_output_bias(self, rewards: list[BinaryReward]) -> None:
        """Set the output-layer bias mu to ``logit(empirical_reward_rate)`` on the first update call.

        Replaces the cold-start prior mean (logit 0 ≈ 50 % reward rate) with a data-driven intercept
        derived from the observed reward rate in ``rewards``.  This prevents the model from
        over-exploring arms that have accumulated little data but whose prior pushes predicted
        probability toward 0.5, making them appear more rewarding than they actually are.

        The calibration fires exactly once per arm lifetime (or after a ``_reset()`` call).
        Subsequent calls are no-ops guarded by the ``bias_calibrated`` flag.

        Parameters
        ----------
        rewards : list[BinaryReward]
            Binary rewards (0/1) observed in the current update batch.  The empirical reward rate
            (mean of ``rewards``) is clipped to ``[_numerical_eps, 1 - _numerical_eps]`` before
            the logit transform to avoid ``log(0)`` / ``log(inf)`` instability.

        Notes
        -----
        - The sigma of the output-layer bias prior is left unchanged; only ``mu`` is updated.
        - This method is called from ``_update`` before VI training begins, so the
          calibrated intercept serves as the warm-start location for the variational guide.
        - The clipping bounds use ``_numerical_eps`` to stay safely away from the degenerate logit values at 0 and 1.
        """
        if self.bias_calibrated or len(rewards) == 0:
            return
        reward_rate = float(np.clip(np.mean(rewards), self._numerical_eps, 1 - self._numerical_eps))
        intercept = float(np.log(reward_rate / (1 - reward_rate)))
        output_layer = self.model_params.bnn_layer_params[-1]
        new_mu = np.full(output_layer.bias.shape, intercept)
        new_bias = output_layer.bias.with_dist_parameters(mu=new_mu.tolist())
        self.model_params.bnn_layer_params[-1] = BnnLayerParams(weight=output_layer.weight, bias=new_bias)
        self.bias_calibrated = True

    def _reset(self):
        """
        Reset the model to its initial parameters.
        """
        self.model_params.bnn_layer_params = deepcopy(self.model_params.bnn_layer_params_init)
        if self.model_params.embedding_params is not None:
            self.model_params.embedding_params.embeddings = deepcopy(self.model_params.embedding_params.embeddings_init)
        self.bias_calibrated = False


class BayesianNeuralNetwork(BaseBayesianNeuralNetwork):
    """
    Bayesian Neural Network class.
    This class implements a Bayesian Neural Network by extending the
    BaseBayesianNeuralNetwork. It provides functionality for probabilistic
    modeling and inference using neural networks.
    """


class BayesianNeuralNetworkCC(BaseBayesianNeuralNetwork, ModelCC):
    """Bayesian Neural Network model for binary classification with cost constraint.

    This class implements a Bayesian Neural Network with an arbitrary number of fully connected layers
    using NumPyro for binary classification tasks. Posterior inference is by Variational Inference (VI).

    Parameters
    ----------
    model_params : BnnParams
        The parameters of the Bayesian Neural Network, including weights and biases for each layer and their initial values for resetting
    update_kwargs : dict | None, optional
        A dictionary of keyword arguments for VI training.
    cost : NonNegativeFloat
        Cost associated to the Bayesian Neural Network model.

    Notes
    -----
    - The model uses tanh activation for hidden layers and sigmoid activation for the output layer.
    - The output layer is designed for binary classification tasks, with probabilities modeled
      using a Bernoulli likelihood.
    """


class BayesianNeuralNetworkDP(BaseBayesianNeuralNetwork, ModelDP):
    """Bayesian Neural Network model for binary classification with dynamic pricing.

    This class implements a Bayesian Neural Network with an arbitrary number of fully connected layers
    using NumPyro for binary classification tasks. Posterior inference is by Variational Inference (VI).

    Parameters
    ----------
    model_params : BnnParams
        The parameters of the Bayesian Neural Network, including weights and biases for each layer and their initial values for resetting
    update_kwargs : dict | None, optional
        A dictionary of keyword arguments for VI training.
    price : NonNegativeFloat
        Price associated to the Bayesian Neural Network model.

    Notes
    -----
    - The model uses tanh activation for hidden layers and sigmoid activation for the output layer.
    - The output layer is designed for binary classification tasks, with probabilities modeled
      using a Bernoulli likelihood.
    """


class BaseBayesianNeuralNetworkMO(ModelMO, ABC):
    """
    Base class for Bayesian Neural Network with multi-objective.

    Parameters
    ----------
    models : list[BayesianNeuralNetwork]
        The list of Bayesian Neural Network models for each objective.
    """

    models: conlist(BayesianNeuralNetwork, min_length=1)

    def model_post_init(self, __context: Any) -> None:
        """
        Validate that all models have the same number of features.
        """
        n_features = self.models[0].input_dim
        for model in self.models[1:]:
            if model.input_dim != n_features:
                raise ValueError(f"All models must have the same number of features: {model.input_dim} != {n_features}")

    @property
    def input_dim(self) -> PositiveInt:
        """
        Returns the expected input dimension of the model.

        Returns
        -------
        PositiveInt
            The number of input features expected by the model, derived from
            the shape of the weight matrix in the first layer's parameters of the first objective model.
        """
        return self.models[0].input_dim

    @property
    def hidden_dim_list(self) -> list[int]:
        """
        Returns the hidden layer dimensions of the model.

        Returns
        -------
        list[int]
            The output dimension of each layer except the last, derived from
            the shape of the weight matrices in the layer parameters.
        """
        return self.models[0].hidden_dim_list

    @classmethod
    @validate_call
    def cold_start(
        cls,
        n_objectives: PositiveInt,
        n_features: PositiveInt,
        hidden_dim_list: list[PositiveInt] | None = None,
        update_kwargs: dict | None = None,
        dist_type: Literal["normal", "studentt"] = "studentt",
        dist_params: dict[str, float] | None = None,
        activation: ActivationFunctions = "tanh",
        use_residual_connections: bool = False,
        use_layerwise_scaling: bool = False,
        bias_std: PositiveFloat | None = None,
        categorical_features: dict[NonNegativeInt, NonNegativeInt] | None = None,
        decay_factor: PositiveFloat01 | None = None,
        **kwargs,
    ) -> Self:
        """
        Initialize a multi-objective Bayesian Neural Network with a cold start.

        Parameters
        ----------
        n_objectives : PositiveInt
            Number of objectives (models) to create.
        n_features : PositiveInt
            Number of input features for each network.
        hidden_dim_list : list[PositiveInt] | None, optional
            List of dimensions for the hidden layers of each network.
        update_kwargs : dict | None, optional
            Additional keyword arguments for VI training.
        dist_type : Literal["normal", "studentt"]
            Type of distribution to use for priors. Default is "studentt".
        dist_params : dict[str, float] | None, optional
            Initial distribution parameters for the network weights and biases.
        activation : str
            The activation function to use for hidden layers. Supported values are: "tanh", "relu", "sigmoid", "gelu" (default is "tanh").
        use_residual_connections : bool
            Whether to use residual connections in the network (default is False).
        use_layerwise_scaling : bool
            Whether to use layerwise scaling in the network (default is False).
        bias_std : PositiveFloat | None
            If provided, overrides ``sigma`` from ``dist_params`` for all layers' bias priors,
            leaving weight priors untouched. Default is None.
        categorical_features : dict[int, int] | None, optional
            Categorical columns as ``{column_index: cardinality}``, forwarded to each per-objective BNN.
        decay_factor : PositiveFloat01 | None
            Per-update forgetting factor forwarded to each per-objective BNN.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        BayesianNeuralNetworkMO
            A multi-objective BNN with the specified number of objectives.
        """

        models = [
            BayesianNeuralNetwork.cold_start(
                n_features=n_features,
                hidden_dim_list=hidden_dim_list,
                update_kwargs=update_kwargs,
                dist_type=dist_type,
                dist_params_init=dist_params,
                activation=activation,
                use_residual_connections=use_residual_connections,
                use_layerwise_scaling=use_layerwise_scaling,
                bias_std=bias_std,
                categorical_features=categorical_features,
                decay_factor=decay_factor,
            )
            for _ in range(n_objectives)
        ]
        return cls(models=models, **kwargs)


class BayesianNeuralNetworkMO(BaseBayesianNeuralNetworkMO):
    """
    Bayesian Neural Network model for multi-objective.

    Parameters
    ----------
    models : list[BayesianNeuralNetwork]
        The list of Bayesian Neural Network models for each objective.
    """


class BayesianNeuralNetworkMOCC(BaseBayesianNeuralNetworkMO, ModelMO, ModelCC):
    """
    Bayesian Neural Network model for multi-objective with cost control.

    Parameters
    ----------
    models : list[BayesianNeuralNetwork]
        The list of Bayesian Neural Network models for each objective.
    cost : NonNegativeFloat
        Cost associated to the Bayesian Neural Network model.
    """
