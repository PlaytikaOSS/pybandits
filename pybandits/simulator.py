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

import os.path
import random
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from bokeh.core.enums import Palette
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, HoverTool, Legend, Plot, TabPanel
from bokeh.palettes import Category10, Turbo256
from bokeh.plotting import figure
from loguru import logger
from scipy.interpolate import make_interp_spline

from pybandits.base import (
    ActionId,
    BinaryReward,
    Probability,
    PyBanditsBaseModel,
    UnifiedActionId,
)
from pybandits.base_model import BaseModelSO
from pybandits.mab import BaseMab
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
    pydantic_version,
)
from pybandits.quantitative_model import QuantitativeModel
from pybandits.utils import in_jupyter_notebook, visualize_via_bokeh

#                                              context    quantity
DoubleParametricActionProbability = Callable[[np.ndarray, np.ndarray], Probability]
#                                  one of: quantity or context
ParametricActionProbability = Callable[[np.ndarray], Probability]
ProbabilityValue = Union[Probability, ParametricActionProbability, DoubleParametricActionProbability]
ActionProbabilityGroundTruth = Dict[ActionId, ProbabilityValue]


class Simulator(PyBanditsBaseModel, ABC):
    """
    Simulate environment for multi-armed bandits.

    This class performs simulation of Multi-Armed Bandits (MAB). Data are processed in batches of size n>=1.
    Per each batch of simulated samples, the mab selects one action and collects the corresponding simulated reward for
    each sample. Then, prior parameters are updated based on returned rewards from recommended actions.

    Parameters
    ----------
    mab : BaseMab
        MAB model.
    n_updates : PositiveInt, defaults  to 10
        The number of updates (i.e. batches of samples) in the simulation.
    batch_size : PositiveInt, defaults to 100
        The number of samples per batch.
    probs_reward : Optional[Union[ActionProbabilityGroundTruth, Dict[str, ActionProbabilityGroundTruth]]]
        The reward probability for the different actions. If None probabilities are set to 0.5.
        The keys of the dict must match the mab actions_ids, and the quantities are float in the interval [0, 1].
        e.g. probs_reward={"a1 A": [0.6], "a2 B": [0.5], "a3": [0.8]}.
        Note that currently only single-objective reward is supported.
    save : bool, defaults to False
        Boolean flag to save the results.
    path : string, default to ''
        Path where_results are saved if save=True
    file_prefix : string, default to ''
        Prefix for the file name where results are saved.
    random_seed : int, default=None
        Seed for random state. If specified, the model outputs deterministic results.
    verbose :  bool, default=False
        Enable verbose output. If True, detailed logging information about the simulation are provided.
    visualize : bool, default=False
        Enable visualization of the simulation results.
    """

    mab: BaseMab
    n_updates: PositiveInt = 10
    batch_size: PositiveInt = 100
    probs_reward: Optional[Union[ActionProbabilityGroundTruth, Dict[str, ActionProbabilityGroundTruth]]] = None
    save: bool = False
    path: str = ""
    file_prefix: str = ""
    random_seed: Optional[NonNegativeInt] = None
    verbose: bool = False
    visualize: bool = False
    _results: pd.DataFrame = PrivateAttr()
    _base_columns: List[str] = PrivateAttr()
    _cumulative_col_prefix: str = "cum"
    # Define dash patterns, markers, and colors for lines
    _dash_patterns = ["solid", "dashed", "dotted"]
    _markers = ["circle", "square", "triangle", "diamond", "star"]

    ############################################ Instance Input Validators #############################################

    if pydantic_version == PYDANTIC_VERSION_1:

        class Config:
            arbitrary_types_allowed = True
            allow_population_by_field_name = True

    elif pydantic_version == PYDANTIC_VERSION_2:
        model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}
    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    @classmethod
    def _validate_probs_reward_dict(
        cls, action_probability_ground_truth: ActionProbabilityGroundTruth, actions: Dict[ActionId, BaseModelSO]
    ):
        if set(action_probability_ground_truth.keys()) != set(actions.keys()):
            raise ValueError("The keys of the action probability ground truth dictionary must match the actions.")
        for action_id, probability in action_probability_ground_truth.items():
            is_quantitative_action = isinstance(actions[action_id], QuantitativeModel)
            cls._validate_probs_reward_values(probability, is_quantitative_action)

    @classmethod
    @abstractmethod
    def _validate_probs_reward_values(cls, probability: ProbabilityValue, is_quantitative_action: bool):
        pass

    @field_validator("file_prefix", mode="before")
    def maybe_alter_file_prefix(cls, value):
        return f"{value}_" if value else ""

    @model_validator(mode="before")
    @classmethod
    def validate_visualize_without_save(cls, values):
        visualize = cls._get_value_with_default("visualize", values)
        save = cls._get_value_with_default("save", values)
        if visualize and not save and not in_jupyter_notebook():
            raise ValueError("Visualize cannot be enabled without saving the results if shell is not Jupyter.")
        return values

    ####################################################################################################################

    def model_post_init(self, __context: Any) -> None:
        # set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.default_rng(self.random_seed)
        self._initialize_results()

    def with_probs_reward(self, probs_reward: pd.DataFrame) -> "Simulator":
        """
        Set the reward probability for the different actions.

        Parameters
        ----------
        probs_reward : pd.DataFrame
            The reward probability for the different actions. The keys of the dict must match the mab actions_ids,
            and the values are float in the interval [0, 1].
            e.g. probs_reward=pd.DataFrame({"a1 A": [0.6], "a2 B": [0.5], "a3": [0.8]}).
            Note that currently only single-objective reward is supported.

        Returns
        -------
        Simulator
            A new instance of the simulator with the specified reward probabilities.
        """
        return self._with_argument("probs_reward", probs_reward)

    @classmethod
    def _generate_prob_reward(
        cls,
        first_dimension: PositiveInt,
        second_dimension: NonNegativeInt = 0,
        n_points: PositiveInt = 10,
        spline_degree: PositiveInt = 3,
    ) -> Union[ParametricActionProbability, DoubleParametricActionProbability]:
        """
        Generate a spline for the given dimensions.

        Parameters
        ----------
        first_dimension : PositiveInt
            The first dimension.
        second_dimension : NonNegativeInt, defaults to 0
            The second dimension.
        n_points : PositiveInt, defaults to 10
            The number of points to sample
        spline_degree : PositiveInt, defaults to 3
            The degree of the spline.

        Returns
        -------
        Union[ParametricActionProbability, DoubleParametricActionProbability]
            The spline function.
        """
        if spline_degree >= n_points:
            raise ValueError(f"spline_degree ({spline_degree}) must be less than n_points ({n_points})")

        def sigmoid(s: np.ndarray) -> np.ndarray:
            return np.where(s >= 0, 1 / (1 + np.exp(-s)), np.exp(s) / (1 + np.exp(s))).item()

        # Create the spline once

        splines = [
            make_interp_spline(np.linspace(0, 1, n_points), np.random.uniform(-1, 1, n_points), k=spline_degree)
            for _ in range(first_dimension + second_dimension)
        ]
        weights = np.random.uniform(0, 1, first_dimension + second_dimension)
        weights = weights / weights.sum()  # Normalize to sum to 1

        if second_dimension:

            def spline_function(input1: np.ndarray, input2: np.ndarray) -> Probability:
                combined_input = np.concatenate((input1, input2))
                logit = (weights * np.array([spline(x) for spline, x in zip(splines, combined_input)])).sum()
                return sigmoid(logit)

            return spline_function
        else:

            def spline_function(input1: np.ndarray) -> Probability:
                logit = (weights * np.array([spline(x) for spline, x in zip(splines, input1)])).sum()
                return sigmoid(logit)

            return spline_function

    @abstractmethod
    def _initialize_results(self):
        """
        Initialize the results DataFrame. The results DataFrame is used to store the raw simulation results.
        """
        pass

    @property
    def results(self):
        return self._results

    def run(self):
        """
        Start simulation process. It consists in the following steps:
            for i=0 to n_updates
                Consider batch[i] of observation
                mab selects the best action as the action with the highest reward probability to each sample in
                    batch[i].
                Rewards are returned for each recommended action
                Prior parameters are updated based on recommended actions and returned rewards
        """
        for batch_index in range(self.n_updates):
            predict_kwargs, update_kwargs, metadata = self._get_batch_step_kwargs_and_metadata(batch_index)
            self._step(batch_index, metadata, predict_kwargs, update_kwargs)

        self._finalize_results()

        # print results
        if self.verbose:
            self._print_results()

        if self.visualize:
            self._visualize_results()

        # store results
        if self.save:
            if self.verbose:
                logger.info(f"Saving results at {self.path}")
            self._save_results()

    def _step(
        self,
        batch_index: int,
        metadata: Dict[str, List],
        predict_kwargs: Dict[str, Union[int, np.ndarray]],
        update_kwargs: Dict[str, np.ndarray],
    ):
        """
        Perform a step of the simulation process. It consists in the following steps:
            - select actions for batch via mab.predict
            - draw rewards for the selected actions based on metadata according to probs_reward
            - write the selected actions for batch #i in the results matrix
            - update the mab model with the selected actions and the corresponding rewards via mab.update

        Parameters
        ----------
        batch_index : int
            The index of the batch.
        metadata : Dict[str, List]
            The metadata for the selected actions.
        predict_kwargs : Dict[str, Union[int, np.ndarray]]
            Dictionary containing the keyword arguments for the batch used in mab.predict.
        update_kwargs : Dict[str, np.ndarray]
            Dictionary containing the keyword arguments for the batch used in mab.update.
        """
        # select actions for batch #index
        predictions = self.mab.predict(**predict_kwargs)
        actions_quantities = predictions[0]  # location 0 is the actions for both SmabPredictions and CmabPredictions
        actions = [x[0] if isinstance(x, tuple) else x for x in actions_quantities]
        quantities = [x[1] if isinstance(x, tuple) else None for x in actions_quantities]
        if all(q is None for q in quantities):
            quantities = None
        rewards = self._draw_rewards(actions_quantities, metadata, update_kwargs)
        # write the selected actions for batch #i in the results matrix
        batch_results = pd.DataFrame(
            {"action": actions, "reward": rewards, "quantities": quantities, "batch": batch_index, **metadata}
        )
        batch_results = self._finalize_step(batch_results, update_kwargs)
        if not all(col in batch_results.columns for col in self._base_columns):
            raise ValueError(f"The batch results must contain the {self._base_columns} columns")
        self._results = pd.concat((self._results, batch_results), ignore_index=True)
        self.mab.update(actions=actions, rewards=rewards, quantities=quantities, **update_kwargs)

    @staticmethod
    @lru_cache
    def _maximize_prob_reward(
        prob_reward_func: Callable[[np.ndarray], Probability], input_dimension: PositiveInt, n_trials: PositiveInt = 100
    ) -> Probability:
        """
        Maximize the probability of reward for the given function.

        Parameters
        ----------
        prob_reward_func : Callable[[np.ndarray], Probability]
            The probability of reward function.
        input_dimension : PositiveInt
            The input dimension.
        n_trials : PositiveInt, defaults to 100
            The number of otimization trials.

        Returns
        -------
        Probability
            The global maxima of prob_reward_func.
        """

        def objective(trial):
            # Sample points from [0,1] for each dimension
            points = [trial.suggest_float(f"x{i}", 0, 1) for i in range(input_dimension)]
            return prob_reward_func(np.array(points))

        # Configure TPE sampler with multivariate optimization
        sampler = optuna.samplers.TPESampler(
            multivariate=True,  # Enable multivariate optimization
            group=True,  # Sample joint distribution of parameters
            constant_liar=True,  # Better parallel optimization handling
        )

        # Create and configure the study
        study = optuna.create_study(sampler=sampler, direction="maximize")

        # Run optimization
        study.optimize(objective, n_jobs=-1, n_trials=n_trials)  # Use all available cores
        best_value = study.best_value
        if (not isinstance(best_value, float)) or (best_value < 0) or (best_value > 1):
            raise ValueError("The best value must be a float in the interval [0, 1].")
        return best_value

    @abstractmethod
    def _draw_rewards(
        self, actions: List[UnifiedActionId], metadata: Dict[str, List], update_kwargs: Dict[str, np.ndarray]
    ) -> List[BinaryReward]:
        """
        Draw rewards for the selected actions based on metadata according to probs_reward.

        Parameters
        ----------
        actions : List[UnifiedActionId]
            The actions selected by the multi-armed bandit model.
        metadata : Dict[str, List]
            The metadata for the selected actions.
        update_kwargs : Dict[str, np.ndarray]
            Update keyword arguments.

        Returns
        -------
        reward : List[BinaryReward]
            A list of binary rewards.
        """
        pass

    @abstractmethod
    def _extract_ground_truth(self, *args, **kwargs) -> Probability:
        """
        Extract the ground truth probability for the action.

        Returns
        -------
        Probability
            The ground truth probability for the action.
        """
        pass

    @abstractmethod
    def _get_batch_step_kwargs_and_metadata(
        self, batch_index: int
    ) -> Tuple[Dict[str, Union[int, np.ndarray]], Dict[str, np.ndarray], Dict[str, List]]:
        """
        Extract kwargs required for the MAB's update and predict functionality,
        as well as metadata for sample association.

        Parameters
        ----------
        batch_index : int
            The index of the batch.

        Returns
        -------
        predict_kwargs : Dict[str, Union[int, np.ndarray]]
            Dictionary containing the keyword arguments for the batch used in mab.predict.
        update_kwargs : Dict[str, Any]
            Dictionary containing the keyword arguments for the batch used in mab.update.
        metadata : Dict[str, List]
            Dictionary containing the association information for the batch.
        """
        pass

    @abstractmethod
    def _finalize_step(self, batch_results: pd.DataFrame, update_kwargs: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Finalize the step by adding additional information to the batch results.

        Parameters
        ----------
        batch_results : pd.DataFrame
            raw batch results
        update_kwargs : Dict[str, np.ndarray]
            Update keyword arguments

        Returns
        -------
        batch_results : pd.DataFrame
            batch results with added columns
        """
        pass

    def _finalize_results(self):
        """
        Finalize the simulation process. Used to add regret and cumulative regret

        Returns
        -------
        None
        """
        self._results["regret"] = self._results["max_prob_reward"] - self._results["selected_prob_reward"]
        self._results["cum_regret"] = self._results["regret"].cumsum()

    @cached_property
    def _action_ids(self) -> List[ActionId]:
        """
        Get the list of actions.

        Returns
        -------
        List[ActionId]
            The list of actions
        """
        return sorted(list(self.mab.actions.keys()))

    @cached_property
    def _cumulative_actions_cols(self) -> List[str]:
        """
        Get the list of cumulative actions columns.

        Returns
        -------
        : List[str]
            The list of cumulative actions columns
        """
        return [f"{self._cumulative_col_prefix}_{action}" for action in self._action_ids]

    @property
    def _colors(self) -> Palette:
        """
        Get the palette of colors.

        Returns
        -------
        : Palette
            Palette of colors
        """
        n_actions = len(self._action_ids)
        category10_keys = Category10.keys()
        return Category10[max(n_actions, min(category10_keys))] if n_actions <= max(category10_keys) else Turbo256

    @classmethod
    def _impute_missing_counts(cls, df: pd.DataFrame, action_ids: List[ActionId]) -> pd.DataFrame:
        """
        Impute missing counts for actions in the data frame.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame with counts of actions selected by the bandit.
        action_ids : List[ActionId]
            List of action ids.

        Returns
        -------
        df : pd.DataFrame
            Data frame with imputed missing counts for actions.
        """
        for action_id in action_ids:
            if action_id not in df.columns:
                df[action_id] = 0
        return df

    @property
    def selected_actions_count(self) -> pd.DataFrame:
        """
        Get the count of actions selected by the bandit on each batch and
        at the end of the simulation process.

        Returns
        -------
        counts_df : pd.DataFrame
            Data frame with batch serial number as index (or total for all batches), actions as columns,
            and count of recommended actions as quantities
        """
        groupby_cols = [col for col in self._base_columns if col not in ["reward", "action"]]
        counts_df = self._results.groupby(groupby_cols)["action"].value_counts().unstack(fill_value=0).reset_index()
        action_ids = self._action_ids
        counts_df = self._impute_missing_counts(counts_df, action_ids)
        reordered_cols = groupby_cols + action_ids
        counts_df = counts_df[reordered_cols]
        cumulative_actions_cols = self._cumulative_actions_cols
        groupby_cols.remove("batch")
        counts_df[cumulative_actions_cols] = (
            counts_df.groupby(groupby_cols)[action_ids].cumsum() if groupby_cols else counts_df[action_ids].cumsum()
        )
        if groupby_cols:
            grouped_counts_df = self._results.groupby(groupby_cols)["action"].value_counts().unstack().fillna(0)
            grouped_counts_df = self._impute_missing_counts(grouped_counts_df, action_ids)
            grouped_counts_df = grouped_counts_df.assign(batch="total").set_index(["batch"], append=True).reset_index()
            grouped_counts_df[cumulative_actions_cols] = grouped_counts_df[action_ids]
        else:
            grouped_counts_df = pd.DataFrame()
        total_counts_df = counts_df.sum(axis=0).to_frame().T
        total_counts_df = (
            total_counts_df.assign(batch="total", **{col: "total" for col in groupby_cols})
            .set_index(["batch"], drop=True)
            .reset_index()
        )
        total_counts_df[cumulative_actions_cols] = total_counts_df[action_ids]
        counts_df = pd.concat((counts_df, grouped_counts_df, total_counts_df), axis=0, ignore_index=True).set_index(
            groupby_cols + ["batch"], drop=True
        )
        return counts_df

    @property
    def positive_reward_proportion(self) -> pd.DataFrame:
        """
        Get the observed proportion of positive rewards for each a1t the end of the simulation process.

        Returns
        -------
        proportion_df : pd.DataFrame
            Data frame with actions as index, and proportion of positive rewards as quantities
        """
        groupby_cols = [col for col in self._base_columns if col not in ["reward", "batch"]]
        proportion_df = self._results.groupby(groupby_cols)["reward"].mean().to_frame(name="proportion")
        return proportion_df

    def _print_results(self):
        """Private function to print results."""
        logger.info("Simulation results (first 10 observations):\n", self._results.head(10), "\n")
        logger.info("Count of actions selected by the bandit: \n", self.selected_actions_count.iloc[-1], "\n")
        logger.info("Observed proportion of positive rewards for each action:\n", self.positive_reward_proportion, "\n")

    def _save_results(self):
        """Private function to save results."""
        self._results.to_csv(self._get_save_path("simulation_results.csv"), index=False)
        self.selected_actions_count.to_csv(self._get_save_path("selected_actions_count.csv"), index=True)
        self.positive_reward_proportion.to_csv(self._get_save_path("positive_reward_proportion.csv"), index=True)

    def _get_save_path(self, file_name: str) -> str:
        """
        Private function to get the save path.

        Parameters
        ----------
        file_name : str
            The file name.

        Returns
        -------
        full_path : str
            The full path to save the file with attached path and name prefix.
        """
        full_path = os.path.join(self.path, f"{self.file_prefix}{file_name}")
        return full_path

    def _visualize_results(self):
        """Private function to visualize results."""
        action_ids = self._action_ids
        cumulative_actions_cols = self._cumulative_actions_cols
        selected_actions_count = self.selected_actions_count
        selected_actions_rate = 100 * pd.merge(
            selected_actions_count[action_ids].div(selected_actions_count[action_ids].sum(axis=1), axis=0),
            selected_actions_count[cumulative_actions_cols].div(
                selected_actions_count[cumulative_actions_cols].sum(axis=1), axis=0
            ),
            left_index=True,
            right_index=True,
        )
        step_actions_rate = selected_actions_rate[(selected_actions_rate.reset_index().batch != "total").values]
        step_actions_rate = (
            step_actions_rate.unstack(level=list(range(step_actions_rate.index.nlevels)))
            .to_frame("value")
            .reset_index()
        )
        groupby_cols = [col for col in self._base_columns if col not in ["reward", "batch", "action"]]
        grouped_df = (
            step_actions_rate.groupby(groupby_cols if len(groupby_cols) > 1 else groupby_cols[0])
            if groupby_cols
            else [("", step_actions_rate)]
        )

        # plot using bokeh
        tabs = []
        for group_name, rates_df in grouped_df:
            if len(groupby_cols) == 1:
                group_name = (group_name,)
            elif len(groupby_cols) == 0:
                group_name = tuple()
            overall_actions_rate = selected_actions_rate.loc[group_name + ("total",)].to_frame("total").reset_index()
            overall_actions_rate = overall_actions_rate[overall_actions_rate["action"].isin(action_ids)]

            # rate vs step line plot
            step_legend_items = []
            fig_steps = figure(
                title="Selected actions rate across steps",
                x_axis_label="Batch index",
                y_axis_label="Rate [%]",
            )
            for i, action in enumerate(action_ids):
                if action not in sorted(rates_df.action.unique()):
                    continue
                self._add_line_to_figure(fig_steps, step_legend_items, rates_df, i, action)

            self._add_legend_to_figure(step_legend_items, fig_steps)
            fig_steps.add_tools(HoverTool(tooltips=[("batch", "@batch"), ("action", "@action"), ("value", "@value")]))

            # Overall selected actions bars plot
            fig_overall = figure(
                title="Overall selected actions rate",
                x_axis_label="Action",
                y_axis_label="Rate [%]",
                x_range=overall_actions_rate["action"],
            )
            fig_overall.vbar(x="action", top="total", width=0.9, source=ColumnDataSource(overall_actions_rate))
            fig_overall.xgrid.grid_line_color = None
            fig_overall.add_tools(HoverTool(tooltips=[("action", "@action"), ("rate", "@total")]))

            # cumulative rate vs step line plot
            cum_legend_items = []
            fig_cumulative_steps = figure(
                title="Cumulative selected actions rate across steps",
                x_axis_label="Batch index",
                y_axis_label="Rate [%]",
            )
            for i, (action, cum_action) in enumerate(zip(action_ids, cumulative_actions_cols)):
                if action not in rates_df.action.unique():
                    continue
                self._add_line_to_figure(fig_cumulative_steps, cum_legend_items, rates_df, i, action, cum_action)

            self._add_legend_to_figure(cum_legend_items, fig_cumulative_steps)
            fig_cumulative_steps.add_tools(
                HoverTool(tooltips=[("batch", "@batch"), ("action", "@action"), ("value", "@value")])
            )

            tabs.append(
                TabPanel(
                    child=layout(children=[[fig_steps, fig_overall], [fig_cumulative_steps]]),
                    title=f"{'_'.join([str(name_part) for name_part in group_name])}",
                )
            )
        visualize_via_bokeh(self._get_save_path("simulation_results.html"), tabs)

    def _add_line_to_figure(
        self,
        fig: Plot,
        legend_items: List[Tuple[str, List]],
        df: pd.DataFrame,
        index: int,
        action: ActionId,
        action_data_source_id: Optional[str] = None,
    ):
        """
        Add a line corresponding to action based on filtering df using action_data_source_id to the figure.

        Parameters
        ----------
        fig : Plot
            Bokeh figure for which a line should be added.
        legend_items : List[Tuple[str, List]
            List of legend elements, given by tuples of name and associated plot members.
        df : DataFrame
            Data frame to filter for line data.
        index : int
            Line serial number.
        action : ActionId
            Subjected action.
        action_data_source_id : Optional[str], resorts to action if not specified
            Corresponding value to action to filter df by.
        """

        action_data_source_id = action_data_source_id or action

        dash_pattern = self._get_modulus_element(index, self._dash_patterns)
        marker = self._get_modulus_element(index, self._markers)
        color = self._get_modulus_element(index, self._colors)

        action_data = df[df.action == action_data_source_id]
        action_source = ColumnDataSource(action_data)
        line = fig.line("batch", "value", source=action_source, line_width=2, color=color, line_dash=dash_pattern)
        scatter = fig.scatter("batch", "value", source=action_source, size=8, color=color, marker=marker)
        legend_items.append((action, [line, scatter]))

    @staticmethod
    def _add_legend_to_figure(legend_items: List[Tuple[str, List]], fig: Plot):
        """
        Add legend with the legend items to fig.

        Parameters
        ----------
        legend_items : List[Tuple[str, List]
            List of legend elements, given by tuples of name and associated plot members.
        fig : Plot
            Bokeh figure for which a legend should be added.
        """
        legend = Legend(items=legend_items)
        legend.title = "Actions"
        legend.location = "right"
        legend.click_policy = "hide"
        fig.add_layout(legend, "right")

    @staticmethod
    def _get_modulus_element(index: int, elements: List):
        """
        Get the element of the list at the index modulo the length of the list.

        Parameters
        ----------
        index : int
            Required index
        elements : List
            List of elements.

        Returns
        -------
            Element of the list at the index modulo the length of the list
        """
        return elements[index % len(elements)]
