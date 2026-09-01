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
import warnings
from copy import deepcopy
from typing import ClassVar, List, Literal, Optional, Union

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    model_validator,
    validate_call,
)
from typing_extensions import Self

from pybandits.base import (
    PositiveFloat01,
    PyBanditsBaseModel,
)
from pybandits.model.bnn.priors import BaseLocationScaleArray, NormalArray, StudentTArray


class CategoricalFeatureConfig(PyBanditsBaseModel):
    """
    Configuration for a single categorical feature with Bayesian embedding.

    The caller is responsible for pre-encoding categorical values as integer indices
    in the range ``[0, cardinality)``.

    Parameters
    ----------
    column_index : NonNegativeInt
        Column position of this feature in the input numpy array.
    cardinality : PositiveInt
        Number of distinct integer category codes. The context array must contain
        pre-encoded integer indices in the range ``[0, cardinality)``.
    embedding_dim : PositiveInt
        Dimensionality of the embedding vector for this feature. Required; callers going through
        ``cold_start`` get :meth:`DNNMixin.default_categorical_embedding_dim`. Set it to ``cardinality``
        to express a one-hot expansion (an identity embedding), as :class:`MLPBackbone` does.
    """

    column_index: NonNegativeInt
    cardinality: PositiveInt
    embedding_dim: PositiveInt


class FeaturesConfig(PyBanditsBaseModel):
    """
    Specification of the structure of a numpy context array.

    Columns can appear in any order. Categorical features are identified by their
    explicit ``column_index``; all remaining columns are treated as numerical.

    Parameters
    ----------
    n_features : int
        Total number of columns in the input numpy array. Default 0.
    categorical_features_configs : List[CategoricalFeatureConfig]
        List of categorical feature configurations.
    """

    n_features: NonNegativeInt = 0
    categorical_features_configs: List[CategoricalFeatureConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _validate_categorical_columns(cls, values):
        if not isinstance(values, dict):
            return values
        cat_configs = values.get("categorical_features_configs", [])
        n_features = values.get("n_features", 0)
        col_indices = [cfg["column_index"] if isinstance(cfg, dict) else cfg.column_index for cfg in cat_configs]
        if len(col_indices) != len(set(col_indices)):
            raise ValueError("Duplicate column_index values found in categorical_features_configs.")
        for col_index in col_indices:
            if col_index >= n_features:
                raise ValueError(f"column_index {col_index} is out of range for n_features={n_features}.")
        return values

    @property
    def numerical_indices(self) -> List[int]:
        """Sorted list of column positions treated as numerical (not used by any categorical)."""
        cat_cols = {cfg.column_index for cfg in self.categorical_features_configs}
        return [i for i in range(self.n_features) if i not in cat_cols]

    @property
    def n_numerical(self) -> int:
        """Number of numerical columns (derived: n_features minus categorical count)."""
        return self.n_features - len(self.categorical_features_configs)

    @property
    def has_categorical(self) -> bool:
        """True if at least one categorical feature is configured."""
        return bool(self.categorical_features_configs)

    @property
    def total_output_dim(self) -> int:
        """
        Total dimensionality of the concatenated vector fed into the first BNN layer.

        = n_numerical + sum(cat.embedding_dim)
        """
        return self.n_numerical + sum(cfg.embedding_dim for cfg in self.categorical_features_configs)


class EmbeddingParams(PyBanditsBaseModel):
    """
    Stores Bayesian embedding matrices for all categorical features.

    Each embedding matrix has shape ``(cardinality, embedding_dim)`` and is stored as a
    ``BaseLocationScaleArray`` (StudentTArray or NormalArray) — the same representation
    used for layer weights in ``BnnLayerParams``.

    Parameters
    ----------
    embeddings : List[Union[StudentTArray, NormalArray]]
        Ordered list of embedding matrix distributions, matching the order of
        ``FeaturesConfig.categorical_features_configs``.
        Shape of each matrix: ``(cardinality, embedding_dim)``.
    embeddings_init : List[Union[StudentTArray, NormalArray]]
        Frozen copy of the initial embeddings for resetting. Set automatically.
    """

    embeddings: List[Union[StudentTArray, NormalArray]]
    embeddings_init: List[Union[StudentTArray, NormalArray]] = Field(default_factory=list, init=False, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _set_embeddings_init(cls, values):
        if isinstance(values, dict):
            if not values.get("embeddings_init"):
                values["embeddings_init"] = deepcopy(values.get("embeddings", []))
        return values

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def cold_start(
        cls,
        feature_config: "FeaturesConfig",
        dist_class: type[BaseLocationScaleArray] = StudentTArray,
        **dist_params_init,
    ) -> Self:
        """
        Create ``EmbeddingParams`` with prior distributions for all categorical features.

        Parameters
        ----------
        feature_config : FeaturesConfig
            Feature configuration containing categorical feature specs.
        dist_class : type
            The distribution class to use for embedding priors, by default ``StudentTArray``.
        **dist_params_init
            Distribution parameters passed to ``BaseLocationScaleArray.cold_start``
            (e.g. ``mu``, ``sigma``, ``nu`` for StudentT; ``mu``, ``sigma`` for Normal).

        Returns
        -------
        EmbeddingParams
            An ``EmbeddingParams`` instance with one embedding matrix per categorical feature,
            each of shape ``(cardinality, embedding_dim)``, initialised from ``dist_class`` cold-start priors.
        """
        embeddings = []
        for cat_cfg in feature_config.categorical_features_configs:
            shape = (cat_cfg.cardinality, cat_cfg.embedding_dim)
            embeddings.append(dist_class.cold_start(shape=shape, **dist_params_init))
        return cls(embeddings=embeddings)


class BnnLayerParams(PyBanditsBaseModel):
    """
    Represents the parameters of a Bayesian neural network (BNN) layer.

    Parameters
    ----------
    weight : Union[NormalArray, StudentTArray]
        The weight parameter of the BNN layer, represented as either a NormalArray or StudentTArray.
    bias : Union[StudentTArray, NormalArray]
        The bias parameter of the BNN layer, represented as either a StudentTArray or NormalArray.
    """

    weight: Union[NormalArray, StudentTArray]
    bias: Union[StudentTArray, NormalArray]


class BnnParams(PyBanditsBaseModel):
    """
    Represents the parameters of a Bayesian Neural Network (BNN), including
    both the current layer parameters and the initial layer parameters.
    We keep the init parameters in case we need to reset the model.

    Parameters
    ----------
    bnn_layer_params : List[BnnLayerParams]
        A list of BNN layer parameters representing the current state of the model.
    bnn_layer_params_init : List[BnnLayerParams]
        A list of BNN layer parameters representing the initial state of the model.
    embedding_params : Optional[EmbeddingParams]
        Bayesian embedding matrices for categorical features. ``None`` when no
        categorical features are configured.
    embedding_params_init : Optional[EmbeddingParams]
        Frozen copy of the initial embedding parameters for resetting. Set automatically.
    """

    bnn_layer_params: Optional[List[BnnLayerParams]]
    bnn_layer_params_init: List[BnnLayerParams] = Field(default_factory=list, init=False, frozen=True)
    embedding_params: Optional[EmbeddingParams] = None
    embedding_params_init: Optional[EmbeddingParams] = Field(default=None, init=False, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, values):
        if values.get("bnn_layer_params_init") is None:
            values["bnn_layer_params_init"] = deepcopy(values["bnn_layer_params"])
        if values.get("embedding_params_init") is None and values.get("embedding_params") is not None:
            values["embedding_params_init"] = deepcopy(values["embedding_params"])

        return values


class EarlyStopping(PyBanditsBaseModel):
    """Early stopping monitor for SVI training.

    Monitors loss convergence and signals when training should stop.
    Stops after ``patience`` consecutive epochs where the loss change is below ``tolerance``.

    Parameters
    ----------
    patience : PositiveInt
        Number of consecutive non-improving epochs required before stopping.
    tolerance : PositiveFloat
        Threshold for convergence.
    diff_type : Literal["relative", "absolute"]
        Type of difference to check: "relative" or "absolute".
    """

    _epsilon: ClassVar[PositiveFloat] = 1e-10

    patience: PositiveInt = 10
    tolerance: PositiveFloat = 1e-4
    diff_type: Literal["relative", "absolute"] = "relative"
    _previous_loss: Optional[float] = PrivateAttr(default=None)
    _no_improvement_count: int = PrivateAttr(default=0)

    def reset(self) -> None:
        """Reset early stopping state for a new training run."""
        self._previous_loss = None
        self._no_improvement_count = 0

    def should_stop(self, loss: float) -> bool:
        """Check if training should stop based on loss convergence."""
        if self._previous_loss is not None:
            if self.diff_type == "relative":
                change = abs((loss - self._previous_loss) / (abs(self._previous_loss) + self._epsilon))
            elif self.diff_type == "absolute":
                change = abs(loss - self._previous_loss)
            else:
                raise ValueError(f"Unknown diff {self.diff_type}")

            if change < self.tolerance:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
        self._previous_loss = loss
        return self._no_improvement_count >= self.patience


class VIUpdateKwargs(PyBanditsBaseModel):
    """Validated keyword arguments for a Variational Inference (VI) BNN update.

    Replaces the previously untyped ``update_kwargs`` dict (the BNN is VI-only).
    Constrained fields validate the values that used to be checked by hand; the nested
    ``optimizer_kwargs``, ``lr_scheduler_kwargs`` and ``early_stopping_kwargs`` dicts are
    intentionally left as open dicts because they are passed through verbatim to optax /
    the EarlyStopping monitor.

    Parameters
    ----------
    num_steps : PositiveInt
        Total number of SVI steps. Ignored when ``epochs`` is provided. Default 1000.
    method : Literal["advi", "fullrank_advi"]
        Variational family / guide. Default "advi".
    optimizer_type : str
        Name of the optax optimizer (resolved at construction). Default "sgd".
    optimizer_kwargs : dict
        Keyword arguments forwarded to the optax optimizer (e.g. ``step_size``).
    batch_size : Optional[PositiveInt]
        Mini-batch size; ``None`` uses the full dataset. Default None.
    early_stopping_kwargs : Optional[dict]
        Keyword arguments forwarded to ``EarlyStopping``; ``None`` disables it. Default None.
    lr_scheduler_type : Optional[str]
        Name of the optax learning-rate schedule, or ``None``. Default None.
    lr_scheduler_kwargs : Optional[dict]
        Keyword arguments forwarded to the optax schedule. Default None.
    restore_best_svi_state : bool
        Whether to restore the lowest-loss SVI state at the end of training. Default True.
    num_particles : PositiveInt
        Number of ELBO particles. Default 1.
    gradient_clip_norm : Optional[PositiveFloat]
        Global gradient-norm clipping threshold, or ``None`` to disable. Default None.
    kl_annealing_fraction : Optional[PositiveFloat01]
        Fraction of total steps over which the KL term is linearly warmed up; must lie in
        the half-open interval (0, 1] or be ``None``. Default None.
    epochs : Optional[PositiveInt]
        Number of epochs; takes precedence over ``num_steps`` when provided. Default None.
    """

    num_steps: PositiveInt = 1000
    method: Literal["advi", "fullrank_advi"] = "advi"
    optimizer_type: str = "adam"
    optimizer_kwargs: dict = Field(default_factory=lambda: {"step_size": 0.0003})
    batch_size: Optional[PositiveInt] = None
    early_stopping_kwargs: Optional[dict] = None
    lr_scheduler_type: Optional[str] = None
    lr_scheduler_kwargs: Optional[dict] = None
    restore_best_svi_state: bool = True
    num_particles: PositiveInt = 1
    gradient_clip_norm: Optional[PositiveFloat] = None
    kl_annealing_fraction: Optional[PositiveFloat01] = None
    epochs: Optional[PositiveInt] = None

    @model_validator(mode="before")
    @classmethod
    def _warn_epochs_and_num_steps(cls, data):
        # Warn only when the user genuinely supplied both — epochs takes precedence over num_steps.
        # Inspecting the raw input (rather than model_fields_set) avoids a spurious warning on
        # round-trip, where model_dump() re-emits epochs=None alongside num_steps as explicit keys.
        if isinstance(data, dict) and data.get("epochs") is not None and "num_steps" in data:
            warnings.warn(
                "Both 'epochs' and 'num_steps' specified in update_kwargs. "
                "'epochs' takes precedence and 'num_steps' will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return data
