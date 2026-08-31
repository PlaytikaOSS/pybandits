# MIT License
#
# Copyright (c) 2022 Playtika Ltd.
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

from abc import ABC
from typing import Any, Self, get_args

import numpy as np
from numpy.typing import ArrayLike
from pydantic import (
    NonNegativeInt,
    PositiveInt,
    validate_call,
)

from pybandits.base import (
    BinaryReward,
    Probability,
    QuantitativeProbability,
    QuantitativeProbabilityWeight,
    QuantitativeWeight,
)
from pybandits.model import BayesianNeuralNetwork
from pybandits.quantitative_model.base import QuantitativeModel, QuantitativeModelCC, QuantitativeModelDP


class BaseQuantitativeBayesianNeuralNetwork(QuantitativeModel, ABC):
    """
    A Bayesian Neural Network based QuantitativeModel.

    This class implements a quantitative model using a Bayesian Neural Network
    where quantities are used as input features to predict reward probabilities.
    The BNN learns the relationship between quantities and rewards.

    Parameters
    ----------
    dimension: PositiveInt
        Number of quantity dimensions (input features for the BNN).
    bnn: BayesianNeuralNetwork
        The underlying Bayesian Neural Network model.
    """

    bnn: BayesianNeuralNetwork

    @classmethod
    @validate_call
    def cold_start(
        cls,
        dimension: PositiveInt = 1,
        n_features: NonNegativeInt = 1,
        categorical_features: dict[NonNegativeInt, NonNegativeInt] | None = None,
        base_model_cold_start_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a cold start QuantitativeBayesianNeuralNetwork model.

        Parameters
        ----------
        dimension : PositiveInt
            Dimension of the quantity (action) space. Default is 1.
        n_features : NonNegativeInt
            Total number of columns in the context array, including any categorical columns.
            Default is 1.
        categorical_features : dict[NonNegativeInt, NonNegativeInt] | None
            Categorical context columns as ``{column_index: cardinality}``.
        base_model_cold_start_kwargs : dict[str, Any] | None, optional
            Keyword arguments passed to BayesianNeuralNetwork.cold_start. May include e.g.
            hidden_dim_list, update_kwargs, dist_type, dist_params_init,
            activation, use_residual_connections, use_layerwise_scaling, decay_factor. Default is None.
        **kwargs
            Additional keyword arguments for the QuantitativeBayesianNeuralNetwork constructor.

        Returns
        -------
        Self
            A cold start QuantitativeBayesianNeuralNetwork model.
        """
        # Quantity columns occupy positions 0..dimension-1 in the BNN input.
        # Shift every context-frame column index by `dimension` so it points at the
        # correct position in the combined [quantity, context] array.
        bnn_categorical = (
            {col_idx + dimension: cardinality for col_idx, cardinality in categorical_features.items()}
            if categorical_features
            else None
        )
        base_model_cold_start_kwargs = dict(base_model_cold_start_kwargs or {})
        bnn = BayesianNeuralNetwork.cold_start(
            n_features=dimension + n_features,
            categorical_features=bnn_categorical,
            **base_model_cold_start_kwargs,
        )

        return cls(
            dimension=dimension,
            bnn=bnn,
            **kwargs,
        )

    @property
    def input_dim(self) -> PositiveInt:
        """
        Returns the expected context dimension of the model (number of context columns).

        Returns
        -------
        PositiveInt
            The number of context columns expected by the model, i.e.
            ``feature_config.n_features - dimension``.
        """
        return self.bnn.feature_config.n_features - self.dimension

    def _to_quantitative_probabilities(
        self,
        context: np.ndarray,
        sampled_weights: list[list[tuple[np.ndarray, np.ndarray]]],
        sampled_embeddings: list[np.ndarray] | None = None,
    ) -> list[QuantitativeProbabilityWeight]:
        """
        Convert the sampled weights to quantitative probabilities and weights.

        Parameters
        ----------
        context : np.ndarray
            The context at which to evaluate the probability.
        sampled_weights : list[list[tuple[np.ndarray, np.ndarray]]]
            The sampled weights.
        sampled_embeddings : list[np.ndarray] | None
            Pre-sampled embedding vectors, one per categorical feature, each of shape
            ``(n_samples, emb_dim)``.  ``None`` when no categorical features are configured.
            Passed as-is to ``forward_pass`` to guarantee a deterministic forward pass per
            sample — the embeddings are fixed for all quantity evaluations of that sample.
        """
        n_samples = len(context)
        # QuantitativeProbabilityWeight is a tuple[QuantitativeProbability, QuantitativeWeight];
        # the number of network outputs equals the number of elements in that tuple.
        n_outputs = len(get_args(QuantitativeProbabilityWeight))

        result = []
        for sample_idx in range(n_samples):
            weights_idx, emb_idx = self.bnn.extract_sample(sampled_weights, sampled_embeddings, sample_idx)

            def create_probability_or_weight_function(
                sample_idx: NonNegativeInt,
                output_index: NonNegativeInt,
                weights_idx=weights_idx,
                emb_idx=emb_idx,
            ) -> QuantitativeProbability | QuantitativeWeight:
                def probability_or_weight_function(quantity: float | np.ndarray) -> Probability | float:
                    bnn_input = self._prepare_network_input([quantity], context[sample_idx])
                    return self.bnn.forward_pass(
                        sampled_weights=weights_idx,
                        context=bnn_input,
                        sampled_embeddings=emb_idx,
                    )[0][output_index]

                return probability_or_weight_function

            result.append(
                tuple(
                    create_probability_or_weight_function(sample_idx, output_index) for output_index in range(n_outputs)
                )
            )
        return result

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def sample_proba(self, context: np.ndarray, rng: np.random.Generator) -> list[QuantitativeProbabilityWeight]:
        """
        Create probability functions which receive the context and creates a function that evaluates the probability given a quantity for each sample.

        Parameters
        ----------
        context : np.ndarray
            The context at which to evaluate the probability.
        rng : np.random.Generator
            Numpy random generator forwarded to BNN weight/embedding sampling.

        Returns
        -------
        list[QuantitativeProbabilityWeight]
            A list of (probability, weight) callables per sample, each taking a quantity (float | np.ndarray).
        """
        _context = np.atleast_2d(context)
        n_samples = _context.shape[0]

        # Pre-sample weights and embeddings so the forward pass is deterministic.
        sampled_weights = self.bnn.sample_weights(n_samples, rng=rng)
        # Build a dummy full array so sample_embeddings can extract the right columns from the context part.
        dummy_full = np.column_stack([np.zeros((n_samples, self.dimension)), _context])
        sampled_embeddings = self.bnn.sample_embeddings(dummy_full, rng=rng)

        result = self._to_quantitative_probabilities(
            context=_context,
            sampled_weights=sampled_weights,
            sampled_embeddings=sampled_embeddings,
        )
        return result

    @staticmethod
    def _prepare_network_input(
        quantity: list[float | np.ndarray | list[float] | tuple[float, ...]], context: ArrayLike
    ) -> np.ndarray:
        """
        Prepare the input for the network, concatenating quantity and context. Quantity can be a float, a 1D array, or a tuple of floats.

        Parameters
        ----------
        quantity : list[float | np.ndarray | list[float] | tuple[float, ...]]
            The quantity value(s) associated with each observation.
            Each element can be a float (for 1D quantities) or a list (for multi-dimensional).
        context : ArrayLike
            The context value(s) associated with each observation.

        Returns
        -------
        np.ndarray
            The input for the network, concatenated quantity and context.
        """
        if len(quantity) == 0:
            raise TypeError("Quantity must be a non-empty 1D array-like value.")
        q0 = quantity[0]
        if isinstance(q0, np.ndarray):
            if q0.ndim != 1:
                raise TypeError("Quantity must be a 1D array-like value.")
        elif isinstance(q0, (list, tuple)):
            if np.asarray(q0).ndim != 1:
                raise TypeError("Quantity must be a 1D array-like value.")
        elif not isinstance(q0, (float, int, np.floating, np.integer)):
            raise TypeError("Quantity must be numeric or a 1D array-like value.")

        _context = np.atleast_2d(context)
        _quantity = np.atleast_2d(quantity).reshape(_context.shape[0], -1)
        return np.concatenate([_quantity, _context], axis=1)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _quantitative_update(
        self,
        quantities: list[float | list[float]],
        rewards: list[BinaryReward],
        context: np.ndarray,
    ):
        """
        Update the BNN model parameters with new quantities and rewards.

        Parameters
        ----------
        quantities : list[float | list[float]]
            The quantity values associated with each observation (None entries are skipped).
        rewards : list[BinaryReward]
            The binary reward for each observation.
        context : np.ndarray
            The context at which to evaluate the probability.
        """
        self._validate_params_lengths(
            True,
            quantities=quantities,
            rewards=rewards,
            context=context,
        )
        _context = np.atleast_2d(context)

        bnn_input = self._prepare_network_input(list(quantities), _context)
        self.bnn.update(context=bnn_input, rewards=rewards)

    def _reset(self):
        """Reset the model to its initial state."""
        self.bnn.reset()


class QuantitativeBayesianNeuralNetwork(BaseQuantitativeBayesianNeuralNetwork):
    """
    A Bayesian Neural Network based QuantitativeModel.

    This class implements a quantitative model using a Bayesian Neural Network
    where quantities are used as input features to predict reward probabilities.
    The BNN learns the relationship between quantities and rewards.

    Parameters
    ----------
    dimension : PositiveInt
        Number of quantity dimensions (input features for the BNN).
    bnn : BayesianNeuralNetwork
        The underlying Bayesian Neural Network model.
    hidden_dim_list : list[PositiveInt] | None
        List of hidden layer dimensions for the BNN. None means no hidden layers.
    update_kwargs : dict | None
        Additional keyword arguments for the update method.

    Examples
    --------
    >>> # Create a cold start model with 2 quantity dimensions
    >>> model = QuantitativeBayesianNeuralNetwork.cold_start(
    ...     dimension=2,
    ...     hidden_dim_list=[8, 4],
    ... )
    >>> # Sample probability functions (context required for BNN)
    >>> context = np.zeros((3, 1))  # (n_samples, n_features)
    >>> prob_funcs = model.sample_proba(context=context, rng=np.random.default_rng())
    >>> # Evaluate probability at a specific quantity
    >>> prob, weight = prob_funcs[0]
    >>> prob_at_q = prob(np.array([0.3, 0.7]))
    >>> # Update with observations
    >>> quantities = [[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]]
    >>> rewards = [1, 0, 1]
    >>> model._quantitative_update(quantities, rewards, context=context)
    """


class QuantitativeBayesianNeuralNetworkCC(BaseQuantitativeBayesianNeuralNetwork, QuantitativeModelCC):
    """
    A Bayesian Neural Network based QuantitativeModel with cost control.

    This class extends QuantitativeBayesianNeuralNetwork with cost control functionality,
    allowing the model to incorporate cost considerations when making decisions.

    Parameters
    ----------
    dimension : PositiveInt
        Number of quantity dimensions (input features for the BNN).
    bnn : BayesianNeuralNetwork
        The underlying Bayesian Neural Network model.
    hidden_dim_list : list[PositiveInt] | None
        List of hidden layer dimensions for the BNN. None means no hidden layers.
    update_kwargs : dict | None
        Additional keyword arguments for the update method.
    cost : Callable[[float | NonNegativeFloat], NonNegativeFloat]
        Cost function that takes a quantity value and returns the associated cost.

    Examples
    --------
    >>> # Create a cold start model with cost control
    >>> model = QuantitativeBayesianNeuralNetworkCC.cold_start(
    ...     dimension=1,
    ...     hidden_dim_list=[4],
    ...     cost=lambda x: x * 0.1  # Linear cost function
    ... )
    """


class QuantitativeBayesianNeuralNetworkDP(BaseQuantitativeBayesianNeuralNetwork, QuantitativeModelDP):
    """
    A Bayesian Neural Network based QuantitativeModel with dynamic pricing.

    This class extends QuantitativeBayesianNeuralNetwork with dynamic pricing functionality,
    allowing the model to incorporate price considerations when making decisions.

    Parameters
    ----------
    dimension : PositiveInt
        Number of quantity dimensions (input features for the BNN).
    bnn : BayesianNeuralNetwork
        The underlying Bayesian Neural Network model.
    hidden_dim_list : list[PositiveInt] | None
        List of hidden layer dimensions for the BNN. None means no hidden layers.
    update_kwargs : dict | None
        Additional keyword arguments for the update method.
    price : Callable[[float | np.ndarray], NonNegativeFloat]
        Price function that takes a quantity value or array and returns the associated price.

    Examples
    --------
    >>> # Create a cold start model with dynamic pricing
    >>> model = QuantitativeBayesianNeuralNetworkDP.cold_start(
    ...     dimension=1,
    ...     hidden_dim_list=[4],
    ...     price=lambda x: x * 10.0  # Linear price function
    ... )
    """
