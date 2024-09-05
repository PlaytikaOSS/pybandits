import os
from copy import deepcopy
from functools import partial
from itertools import product
from math import floor
from multiprocessing import Pool, cpu_count
from sys import version_info
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import optuna
import pandas as pd
from bokeh.models import ColumnDataSource, TabPanel
from bokeh.plotting import figure
from loguru import logger
from optuna import Trial
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
    pydantic_version,
    validate_call,
)

if version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pybandits import offline_policy_estimator
from pybandits.base import ActionId, Float01, PyBanditsBaseModel
from pybandits.mab import BaseMab
from pybandits.offline_policy_estimator import BaseOfflinePolicyEstimator
from pybandits.utils import (
    extract_argument_names_from_function,
    get_non_abstract_classes,
    in_jupyter_notebook,
    visualize_via_bokeh,
)

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


class _FunctionEstimator(PyBanditsBaseModel, ClassifierMixin, arbitrary_types_allowed=True):
    """
    This class provides functionality for model optimization using hyperparameter tuning via Optuna,
    and prediction with optimized or default machine learning models.
    It is used to estimate the propensity score and expected reward.


    Parameters
    ----------
    estimator_type : Optional[Literal["logreg", "gbm", "rf", "mlp"]]
        The model type to optimize.

    fast_fit : bool
        Whether to use the default parameter set for the model.

    action_one_hot_encoder : OneHotEncoder
        Fitted one hot encoder for action encoding.

    n_trials : int
        Number of trials for the Optuna optimization process.

    verbose : bool
        Whether to log detailed information during the optimization process.

    study_name : Optional[str]
        Name of the study to be created by Optuna.

    multi_action_prediction : bool
        Whether to predict for all actions or only for real action.

    """

    estimator_type: Literal["logreg", "gbm", "rf", "mlp"]
    fast_fit: bool
    action_one_hot_encoder: OneHotEncoder = OneHotEncoder(sparse=False)
    n_trials: int
    verbose: bool
    study_name: Optional[str] = None
    multi_action_prediction: bool
    _model: Union[LogisticRegression, GradientBoostingClassifier, RandomForestClassifier, MLPClassifier] = PrivateAttr()
    _model_mapping = {
        "mlp": MLPClassifier,
        "rf": RandomForestClassifier,
        "logreg": LogisticRegression,
        "gbm": GradientBoostingClassifier,
    }

    def _pre_process(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess the feature vectors to be used for regression model training.
        This method concatenates the context vector and action context vectors.

        Parameters
        ----------
        batch : Dict[str, Any]
            The batch of data containing context, action, and action context.

        Returns
        -------
        np.ndarray
            A concatenated array of context and action context, shape (n_rounds, n_features_context + dim_action_context).
        """
        context = batch["context"]
        action = batch["action_ids"]
        return np.concatenate([context, self.action_one_hot_encoder.transform(action.reshape((-1, 1)))], axis=1)

    def _sample_parameter_space(self, trial: Trial) -> Dict[str, Union[str, int, float]]:
        """
        Define the hyperparameter search space for a given model type in Optuna.

        The search space is dynamically selected based on the model type being optimized.

        Parameters
        ----------
        trial : optuna.Trial
            A single trial in the Optuna optimization process.

        Returns
        -------
        dict
            A dictionary representing the search space for the model's hyperparameters.
        """

        if self.estimator_type == "mlp":
            return {
                "hidden_layer_sizes": 2 ** trial.suggest_int("hidden_layer_sizes", 2, 6),
                "activation": trial.suggest_categorical("activation", ["relu", "logistic", "tanh"]),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
                "alpha": np.sqrt(10) ** -trial.suggest_int("learning_rate_init", 0, 10),
                "max_iter": 1000,
                "learning_rate_init": np.sqrt(10) ** -trial.suggest_int("learning_rate_init", 0, 6),
            }
        elif self.estimator_type == "rf":
            return {
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_features": trial.suggest_int("max_features", 1, 3),
                "n_estimators": trial.suggest_int("n_estimators", 10, 50),
                "n_jobs": -1,
            }
        elif self.estimator_type == "logreg":
            return {
                "tol": trial.suggest_float("tol", 0.00001, 0.0001),
                "C": trial.suggest_float("C", 0.05, 3),
                "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
                "max_iter": 1000,
                "n_jobs": -1,
            }
        elif self.estimator_type == "gbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "learning_rate": np.sqrt(10) ** -trial.suggest_int("learning_rate_init", 0, 6),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
            }

    def _objective(self, trial: Trial, feature_set: np.ndarray, label: np.ndarray) -> float:
        """
        Objective function for Optuna optimization.

        This function trains a model using cross-validation and returns the negative accuracy
        to be minimized.

        Parameters
        ----------
        trial : Trial
            A single trial in the Optuna optimization process.

        feature_set : np.ndarray
            The training dataset, containing context and encoded actions.

        label : np.ndarray
            The labels for the dataset.

        Returns
        -------
        score : float
            The score to be maximized by Optuna.
        """
        params = self._sample_parameter_space(trial)
        model = self._model_mapping[self.estimator_type](**params)
        score = cross_val_score(model, feature_set, label).mean()
        trial.set_user_attr("model_params", params)

        return score

    def _optimize(self, feature_set: np.ndarray, label: np.ndarray, study: optuna.Study) -> dict:
        """
        Optimize the model's hyperparameters using Optuna.

        Parameters
        ----------
        feature_set : np.ndarray
            The training dataset, containing 'context' and 'action_ids' keys.

        study : optuna.Study
            The Optuna study object to store optimization results.

        Returns
        -------
        best_params : dict
            The best set of hyperparameters found by Optuna.
        """

        study.optimize(lambda trial: self._objective(trial, feature_set, label), n_trials=self.n_trials)

        best_params = study.best_trial.user_attrs["model_params"]
        if self.verbose:
            logger.info(f"Optuna best model with optimized parameters for {self.estimator_type}:\n {best_params}")

        return best_params

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: dict, y: np.ndarray) -> Self:
        """
        Fit the model using the given dataset X and labels y.

        Parameters
        ----------
        X : dict
            The dataset containing 'context' and 'action_ids' keys.
        y : np.ndarray
            The labels for the dataset.

        Returns
        -------
        self : _FunctionEstimator
            The fitted model.
        """
        feature_set = self._pre_process(X)
        if self.fast_fit:
            model_parameters = {}
        else:
            pruner = optuna.pruners.MedianPruner()
            sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
            study = optuna.create_study(
                direction="maximize", study_name=self.study_name, pruner=pruner, sampler=sampler
            )
            model_parameters = self._optimize(feature_set, y, study)

        model = self._model_mapping[self.estimator_type](**model_parameters)
        model.fit(feature_set, y)
        self._model = model
        return self

    @validate_call
    def predict(self, X: dict) -> np.ndarray:
        """
        Predict the labels for the given dataset X.

        Parameters
        ----------
        X : dict
            The dataset containing 'context' and 'action_ids' keys.

        Returns
        -------
        prediction : np.ndarray
            The predicted labels for the dataset.
        """
        if not self._model:
            raise AttributeError("Model has not been fitted yet.")

        if self.multi_action_prediction:
            specific_action_X = X.copy()
            prediction = np.empty((X["n_rounds"], len(X["unique_actions"])))
            for action_index, action in enumerate(X["unique_actions"]):
                specific_action_X["action_ids"] = np.array([action] * X["n_rounds"])
                specific_action_feature_set = self._pre_process(specific_action_X)
                specific_action_prediction = self._model.predict_proba(specific_action_feature_set)[:, 1]
                prediction[:, action_index] = specific_action_prediction
        else:
            feature_set = self._pre_process(X)
            prediction = self._model.predict_proba(feature_set)[:, 1]
        return prediction


class OfflinePolicyEvaluator(PyBanditsBaseModel, arbitrary_types_allowed=True):
    """
    Class to conduct OPE with multiple OPE estimators

    Reference: Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation
    https://arxiv.org/abs/2008.07146 https://github.com/st-tech/zr-obp

    Parameters
    ----------
    logged_data : pd.DataFrame
        Logging data set
    split_prop: Float01
        Proportion of dataset used as training set
    propensity_score_model_type: Literal["logreg", "gbm", "rf", "mlp", "batch_empirical", "empirical", "propensity_score"]
        Method used to compute/estimate propensity score pi_b (propensity_score, logging / behavioral policy).
    expected_reward_model_type: Literal["logreg", "gbm", "rf", "mlp"]
        Method used to estimate expected reward for each action a in the training set.
    n_trials : Optional[int]
        Number of trials for the Optuna optimization process.
    fast_fit : bool
        Whether to use the default parameter set for the function estimator models.
    ope_estimators: Optional[List[BaseOfflinePolicyEstimator]]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        All available estimators are if not specified.
    batch_feature: str
        Column name for batch as available in logged_data
    action_feature: str
        Column name for action as available in logged_data
    reward_feature: Union[str, List[str]]
        Column name for reward as available in logged_data
    contextual_features: Optional[List[str]]
        Column names for contextual features as available in logged_data
    cost_feature: Optional[str]
        Column name for cost as available in logged_data; used for bandit with cost control
    group_feature: Optional[str]
        Column name for group definition feature as available in logged_data; available from simulated data
        to define samples with similar contextual profile
    true_reward_feature: Optional[Union[str, List[str]]]
        Column names for reward proba distribution features as available in simulated logged_data. Used to compute ground truth
    propensity_score_feature : Optional[str]
        Column name for propensity score as available in logged_data; used for evaluation of the policy value
    verbose : bool
        Whether to log detailed information during the optimization process.
    """

    logged_data: pd.DataFrame
    split_prop: Float01
    propensity_score_model_type: Literal[
        "logreg", "gbm", "rf", "mlp", "batch_empirical", "empirical", "propensity_score"
    ]
    expected_reward_model_type: Literal["logreg", "gbm", "rf", "mlp"]
    importance_weights_model_type: Literal["logreg", "gbm", "rf", "mlp"]
    scaler: Optional[Union[TransformerMixin, Dict[str, TransformerMixin]]] = None
    n_trials: Optional[int] = 100
    fast_fit: bool = False
    ope_estimators: Optional[List[BaseOfflinePolicyEstimator]]
    batch_feature: str
    action_feature: str
    reward_feature: Union[str, List[str]]
    contextual_features: Optional[List[str]] = None
    cost_feature: Optional[str] = None
    group_feature: Optional[str] = None
    true_reward_feature: Optional[Union[str, List[str]]] = None
    propensity_score_feature: Optional[str] = None
    verbose: bool = False
    _train_data: Optional[Dict[str, Any]] = PrivateAttr()
    _test_data: Optional[Dict[str, Any]] = PrivateAttr()
    _estimated_expected_reward: Optional[Dict[str, np.ndarray]] = PrivateAttr(default=None)
    _action_one_hot_encoder = OneHotEncoder(sparse=False)
    _propensity_score_epsilon = 1e-08

    @field_validator("split_prop", mode="before")
    @classmethod
    def check_split_prop(cls, value):
        if value == 0 or value == 1:
            raise ValueError("split_prop should be strictly between 0 and 1")
        return value

    @field_validator("ope_estimators", mode="before")
    @classmethod
    def populate_ope_metrics(cls, value):
        return (
            value if value is not None else [class_() for class_ in get_non_abstract_classes(offline_policy_estimator)]
        )

    @model_validator(mode="before")
    @classmethod
    def check_batch_feature(cls, values):
        if values["batch_feature"] not in values["logged_data"]:
            raise AttributeError("Batch feature missing from logged data.")
        if not (
            pd.api.types.is_datetime64_ns_dtype(values["logged_data"][values["batch_feature"]])
            or pd.api.types.is_integer_dtype(values["logged_data"][values["batch_feature"]])
        ):
            raise TypeError(f"Column {values['batch_feature']} should be either date or int type")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_action_feature(cls, values):
        if values["action_feature"] not in values["logged_data"]:
            raise AttributeError("Action feature missing from logged data.")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_propensity_score_estimation_method(cls, values):
        if values["propensity_score_model_type"] == "propensity_score":
            if cls._get_value_with_default("propensity_score_feature", values) is None:
                raise ValueError(
                    "Propensity score feature should be defined when using it as propensity_score_model_type"
                )
        return values

    @model_validator(mode="before")
    @classmethod
    def check_reward_features(cls, values):
        reward_feature = values["reward_feature"]
        reward_feature = reward_feature if isinstance(reward_feature, list) else [reward_feature]
        if not all([reward in values["logged_data"] for reward in reward_feature]):
            raise AttributeError("Reward feature missing from logged data.")
        values["reward_feature"] = reward_feature
        if "true_reward_feature" in values:
            true_reward_feature = values["true_reward_feature"]
            true_reward_feature = (
                true_reward_feature
                if isinstance(true_reward_feature, list)
                else [true_reward_feature]
                if true_reward_feature is not None
                else None
            )
            if not all([true_reward in values["logged_data"] for true_reward in true_reward_feature]):
                raise AttributeError("True reward feature missing from logged data.")
            if len(reward_feature) != len(true_reward_feature):
                raise ValueError("Reward and true reward features should have the same length")
            values["true_reward_feature"] = true_reward_feature

        return values

    @model_validator(mode="before")
    @classmethod
    def check_optional_scalar_features(cls, values):
        for feature in [
            "cost_feature",
            "group_feature",
            "propensity_score_feature",
        ]:
            value = cls._get_value_with_default(feature, values)
            if value is not None and value not in values["logged_data"]:
                raise AttributeError(f"{feature} missing from logged data.")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_contextual_features(cls, values):
        value = cls._get_value_with_default("contextual_features", values)
        if value is not None and not set(value).issubset(values["logged_data"].columns):
            raise AttributeError("contextual_features missing from logged data.")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_model_optimization(cls, values):
        n_trials_value = cls._get_value_with_default("n_trials", values)
        fast_fit_value = cls._get_value_with_default("fast_fit", values)

        if (n_trials_value is None or fast_fit_value is None) and values["propensity_score_model_type"] not in [
            "logreg",
            "gbm",
            "rf",
            "mlp",
        ]:
            raise ValueError("The requested propensity score model requires n_trials and fast_fit to be well defined")
        if (n_trials_value is None or fast_fit_value is None) and cls._check_argument_required_by_estimators(
            "expected_reward", values["ope_estimators"]
        ):
            raise ValueError(
                "The requested offline policy evaluation metrics model require estimation of the expected reward. "
                "Thus, n_trials and fast_fit need to be well defined."
            )
        return values

    @classmethod
    def _check_argument_required_by_estimators(cls, argument: str, ope_estimators: List[BaseOfflinePolicyEstimator]):
        """
        Check if argument is required by OPE estimators.

        Parameters
        ----------
        argument : str
            Argument to check if required by OPE estimators.
        ope_estimators : List[BaseOfflinePolicyEstimator]
            List of OPE estimators.

        Returns
        -------
        bool
            True if argument is required by OPE estimators, False otherwise.
        """
        return any(
            [
                argument
                in extract_argument_names_from_function(ope_estimator.estimate_sample_rewards)
                + extract_argument_names_from_function(ope_estimator.estimate_policy_value_with_confidence_interval)
                for ope_estimator in ope_estimators
            ]
        )

    if pydantic_version == PYDANTIC_VERSION_1:

        def __init__(self, **data):
            super().__init__(**data)

            # Extract batches for train and test set
            self._extract_batches()

            # Estimate propensity score in the train and test set
            self._estimate_propensity_score()

            # Estimate expected reward estimator and predict in the test set, when required by OPE estimators
            if self._check_argument_required_by_estimators("expected_reward", self.ope_estimators):
                self._estimate_expected_reward()

    elif pydantic_version == PYDANTIC_VERSION_2:

        def model_post_init(self, __context: Any) -> None:
            # Extract batches for train and test set
            self._extract_batches()

            # Estimate propensity score in the train and test set
            self._estimate_propensity_score()

            # Estimate expected reward estimator and predict in the test set, when required by OPE estimators
            if self._check_argument_required_by_estimators("expected_reward", self.ope_estimators):
                self._estimate_expected_reward()

    else:
        raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

    def _extract_batches(self):
        """
        Create training and test sets in dictionary form.

        """
        logged_data = self.logged_data.sort_values(by=self.batch_feature)
        unique_batch = logged_data[self.batch_feature].unique()
        split_batch = unique_batch[int(floor(len(unique_batch) * self.split_prop))]

        # add list of actions in dict in order to avoid test set with n_actions
        # lower than nb of total actions
        unique_actions = sorted(self.logged_data[self.action_feature].unique().tolist())
        action_label_encoder = LabelEncoder()
        for batch_idx in tqdm(range(2)):
            # extract samples batch
            if batch_idx == 0:
                extracted_batch = self.logged_data[self.logged_data[self.batch_feature] <= split_batch]
            else:
                extracted_batch = self.logged_data[self.logged_data[self.batch_feature] > split_batch]
            extracted_batch = extracted_batch.reset_index(drop=True)

            # dict data set information for OPE
            action_ids = extracted_batch[self.action_feature].values
            if batch_idx == 0:
                self._action_one_hot_encoder.fit(np.array(unique_actions).reshape((-1, 1)))
            reward = extracted_batch[self.reward_feature].values

            # if cost control bandit
            if self.cost_feature is not None:
                cost = extracted_batch[self.cost_feature].values
            else:
                cost = None

            # if contextual information required
            if self.contextual_features is not None:
                if self.scaler is not None:
                    if type(self.scaler) is dict:
                        if batch_idx == 0:
                            x_scale = np.array(
                                pd.concat(
                                    [
                                        self.scaler[feature].fit_transform(np.array(extracted_batch[[feature]]))
                                        for feature in self.contextual_features
                                    ],
                                    axis=1,
                                )
                            )
                        else:
                            x_scale = np.array(
                                pd.concat(
                                    [
                                        self.scaler[feature].transform(np.array(extracted_batch[[feature]]))
                                        for feature in self.contextual_features
                                    ],
                                    axis=1,
                                )
                            )
                    else:
                        if batch_idx == 0:
                            x_scale = self.scaler.fit_transform(np.array(extracted_batch[self.contextual_features]))
                        else:
                            x_scale = self.scaler.transform(np.array(extracted_batch[self.contextual_features]))
                else:
                    x_scale = np.array(extracted_batch[self.contextual_features])
            else:
                x_scale = np.zeros((len(action_ids), 0))  # zero-columns 2d array to allow concatenation later

            # extract data for policy information
            policy_information_cols = [
                self.batch_feature,
                self.action_feature,
            ] + self.reward_feature
            if self.group_feature:
                policy_information_cols.append(self.group_feature)

            policy_information = extracted_batch[policy_information_cols]

            # reward probability distribution as used during simulation process if available
            ground_truth = extracted_batch[self.true_reward_feature] if self.true_reward_feature else None

            # propensity_score may be available from simulation: propensity_score is added to the dict
            propensity_score = (
                extracted_batch[self.propensity_score_feature].values if self.propensity_score_feature else None
            )
            if batch_idx == 0:
                action_label_encoder.fit(unique_actions)
            actions = action_label_encoder.transform(action_ids)

            # Store information in a dictionary as required by obp package
            data_batch = {
                "n_rounds": len(action_ids),  # number of samples
                "n_action": len(unique_actions),  # number of actions
                "unique_actions": unique_actions,  # list of actions in the whole data set
                "action_ids": action_ids,  # action identifiers
                "action": actions,  # encoded action identifiers
                "reward": reward,  # samples' reward
                "propensity_score": propensity_score,  # propensity score, pi_b(a|x), vector
                "context": x_scale,  # the matrix of features i.e. context
                "data": policy_information,  # data array with informative features
                "ground_truth": ground_truth,  # true reward probability for each action and samples, list of list
                "cost": cost,  # samples' action cost for bandit with cost control
            }
            if batch_idx == 0:
                self._train_data = data_batch
            else:
                self._test_data = data_batch

    def _estimate_propensity_score_empirical(
        self, batch: Dict[str, Any], groupby_cols: List[str], inner_groupby_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Empirical propensity score computation based on batches average

        Parameters
        ----------
        batch: Dict[str, Any]
            Dataset dictionary
        groupby_cols : List[str]
            Columns to group by
        inner_groupby_cols : Optional[List[str]]
            Columns to group by after the first groupby

        Returns
        -------
        propensity_score : np.ndarray
            computed propensity score for each of the objectives
        """
        inner_groupby_cols = [] if inner_groupby_cols is None else inner_groupby_cols
        overall_groupby_cols = groupby_cols + inner_groupby_cols
        # number of recommended actions per group and batch
        grouped_data = batch["data"].groupby(overall_groupby_cols)[self.reward_feature[0]].count()

        # proportion of recommended actions per group
        if inner_groupby_cols:
            empirical_distribution = pd.DataFrame(
                grouped_data / grouped_data.groupby(inner_groupby_cols).sum()
            ).reset_index()
        else:
            empirical_distribution = pd.DataFrame(grouped_data / grouped_data.sum()).reset_index()

        empirical_distribution.columns = overall_groupby_cols + ["propensity_score"]

        # deal with missing segment after group by
        if len(overall_groupby_cols) > 1:
            all_combinations = pd.DataFrame(
                list(product(*[empirical_distribution[col].unique() for col in overall_groupby_cols])),
                columns=overall_groupby_cols,
            )

            # Merge with the original dataframe, filling missing values in 'c' with 0
            empirical_distribution = pd.merge(
                all_combinations, empirical_distribution, on=groupby_cols + inner_groupby_cols, how="left"
            ).fillna(0)

        # extract propensity_score in the test set for user according to group and action recommended
        matching_df = pd.DataFrame({k: batch["data"][k] for k in overall_groupby_cols})
        merged_df = pd.merge(
            matching_df,
            empirical_distribution[overall_groupby_cols + ["propensity_score"]],
            how="left",  # left join to ensure we get all rows from the batch
            on=overall_groupby_cols,
        )
        propensity_score = merged_df["propensity_score"].values

        return propensity_score

    def _empirical_averaged_propensity_score(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Empirical propensity score computation based on batches average

        Parameters
        ----------
        batch : Dict[str, Any]
            dataset.

        Returns
        ------
        : np.ndarray
            estimated propensity_score
        """

        return self._estimate_propensity_score_empirical(
            batch=batch, groupby_cols=[self.action_feature], inner_groupby_cols=[self.batch_feature]
        )

    def _empirical_propensity_score(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Propensity score empirical computation based on data set average

        Parameters
        ----------
        batch : Dict[str, Any]
            dataset.

        Return
        ------
        np.ndarray
            estimated propensity_score
        """

        return self._estimate_propensity_score_empirical(batch=batch, groupby_cols=[self.action_feature])

    def _estimate_propensity_score(self):
        """
        Compute/approximate propensity score based on different methods in the train and test set.
        Different approaches may be evaluated when logging policy is unknown.
        """
        if not self.contextual_features:
            # if no contextual features, propensity score is directly defined by the action taken,
            # thus uniformly set to 1
            train_propensity_score = np.ones(self._train_data["n_rounds"])
            test_propensity_score = np.ones(self._test_data["n_rounds"])
            logger.warning(
                f"No contextual features available, "
                f"overriding the requested propensity_score_model_type={self.propensity_score_model_type} "
                f"using uniform propensity score"
            )
        else:
            if self.propensity_score_model_type == "batch_empirical":
                if self.verbose:
                    logger.info("Data batch-empirical estimation of propensity score.")

                # Empirical approach: propensity score pi_b computed as action means per samples batch
                train_propensity_score = self._empirical_propensity_score(self._train_data)
                test_propensity_score = self._empirical_propensity_score(self._test_data)

            elif self.propensity_score_model_type == "empirical":
                if self.verbose:
                    logger.info("Data empirical estimation of propensity score.")

                # Empirical approach: propensity score pi_b computed as action means per samples batch
                train_propensity_score = self._empirical_averaged_propensity_score(self._train_data)
                test_propensity_score = self._empirical_averaged_propensity_score(self._test_data)

            elif self.propensity_score_model_type == "propensity_score":
                if self.verbose:
                    logger.info("Data given value of propensity score.")

                train_propensity_score = self._train_data["propensity_score"]
                test_propensity_score = self._test_data["propensity_score"]

            else:  # self.propensity_score_model_type in ["gbm", "rf", "logreg", "mlp"]
                if self.verbose:
                    logger.info(
                        f"Data prediction of propensity score based on {self.propensity_score_model_type} model."
                    )
                propensity_score_estimator = _FunctionEstimator(
                    estimator_type=self.propensity_score_model_type,
                    fast_fit=self.fast_fit,
                    action_one_hot_encoder=self._action_one_hot_encoder,
                    n_trials=self.n_trials,
                    verbose=self.verbose,
                    study_name=f"{self.propensity_score_model_type}_propensity_score",
                    multi_action_prediction=False,
                )
                propensity_score_estimator.fit(X=self._train_data, y=self._train_data["action"])
                train_propensity_score = np.clip(
                    propensity_score_estimator.predict(self._train_data), self._propensity_score_epsilon, 1
                )
                test_propensity_score = np.clip(
                    propensity_score_estimator.predict(self._test_data), self._propensity_score_epsilon, 1
                )
        self._train_data["propensity_score"] = train_propensity_score
        self._test_data["propensity_score"] = test_propensity_score

    def _estimate_expected_reward(self):
        """
        Compute expected reward for each round and action.
        """
        if self.verbose:
            logger.info(f"Data prediction of expected reward based on {self.expected_reward_model_type} model.")
        estimated_expected_reward = {}
        for reward_feature, reward in zip(self.reward_feature, self._train_data["reward"].T):
            expected_reward_model = _FunctionEstimator(
                estimator_type=self.expected_reward_model_type,
                fast_fit=self.fast_fit,
                action_one_hot_encoder=self._action_one_hot_encoder,
                n_trials=self.n_trials,
                verbose=self.verbose,
                study_name=f"{self.expected_reward_model_type}_expected_reward",
                multi_action_prediction=True,
            )

            expected_reward_model.fit(X=self._train_data, y=reward.T)

            # predict in test set
            estimated_expected_reward[reward_feature] = expected_reward_model.predict(self._test_data)
        self._estimated_expected_reward = estimated_expected_reward

    def _estimate_importance_weight(self, mab: BaseMab) -> np.ndarray:
        """
        Compute importance weights induced by the behavior and evaluation policies.

        Reference: Balanced Off-Policy Evaluation in General Action Spaces (Sondhi, Arbour, and Dimmery, 2020)
                   https://arxiv.org/pdf/1906.03694

        Parameters
        ----------
        mab : BaseMab
            Multi-armed bandit to be evaluated

        Return
        ------
        expected_importance_weights : np.ndarray
            estimated importance weights
        """
        if self.verbose:
            logger.info(f"Data prediction of importance weights based on {self.importance_weights_model_type} model.")

        importance_weights_model = _FunctionEstimator(
            estimator_type=self.importance_weights_model_type,
            fast_fit=self.fast_fit,
            action_one_hot_encoder=self._action_one_hot_encoder,
            n_trials=self.n_trials,
            verbose=self.verbose,
            study_name=f"{self.importance_weights_model_type}_importance_weights",
            multi_action_prediction=False,
        )
        train_data = deepcopy(self._train_data)
        mab_data = self._train_data["context"] if self.contextual_features else self._train_data["n_rounds"]
        selected_actions = _mab_predict(mab, mab_data)
        train_data["action_ids"] = np.concatenate((train_data["action_ids"], selected_actions), axis=0)
        train_data["context"] = np.concatenate((train_data["context"], train_data["context"]), axis=0)
        y = np.concatenate((np.zeros(len(selected_actions)), np.ones(len(selected_actions))), axis=0)
        importance_weights_model.fit(X=train_data, y=y)

        # predict in test set
        estimated_proba = importance_weights_model.predict(self._test_data)
        expected_importance_weights = estimated_proba / (1 - estimated_proba)
        return expected_importance_weights

    def _estimate_policy(
        self,
        mab: BaseMab,
        n_mc_experiments: PositiveInt = 1000,
        n_cores: Optional[NonNegativeInt] = None,
    ) -> np.ndarray:
        """
        Estimate policy via Monte Carlo (MC) sampling based on sampling distribution of each action a in the test set.

        Reference:  Estimation Considerations in Contextual Bandit
                    https://arxiv.org/pdf/1711.07077.pdf
        Reference:  Debiased Off-Policy Evaluation for Recommendation Systems
                    https://arxiv.org/pdf/2002.08536.pdf
        Reference:  CAB: Continuous Adaptive Blending for Policy Evaluation and Learning
                    https://arxiv.org/pdf/1811.02672.pdf

        Parameters
        ----------
        mab : BaseMab
            Multi-armed bandit to be evaluated
        n_mc_experiments: PositiveInt
            Number of MC sampling rounds. Default: 1000
        n_cores: Optional[NonNegativeInt], all available cores if not specified
            Number of cores used for multiprocessing

        Returns
        -------
        estimated_policy: np.ndarray (nb samples, nb actions)
            action probabilities for each action and samples
        """
        if self.verbose:
            logger.info("Data prediction of expected policy based on Monte Carlo experiments.")
        n_cores = n_cores or cpu_count()

        # using MC, create a () best actions matrix
        mc_actions = []
        mab_data = self._test_data["context"] if self.contextual_features else self._test_data["n_rounds"]
        predict_func = partial(_mab_predict, mab, mab_data)
        with Pool(processes=n_cores) as pool:
            # predict best action for a new prior parameters draw
            # using argmax(p(r|a, x)) with a in the list of actions
            for mc_action in tqdm(pool.imap_unordered(predict_func, range(n_mc_experiments))):
                mc_actions.append(mc_action)

        # finalize the dataframe shape to #samples X #mc experiments
        mc_actions = pd.DataFrame(mc_actions).T

        # for each sample / each action, count the occurrence frequency during MC iteration
        estimated_policy = np.zeros((self._test_data["n_rounds"], len(self._test_data["unique_actions"])))
        mc_action_counts = mc_actions.apply(pd.Series.value_counts, axis=1).fillna(0)

        for u in tqdm(range(self._test_data["n_rounds"])):
            estimated_policy[u, :] = (
                mc_action_counts.iloc[u, :].reindex(self._test_data["unique_actions"], fill_value=0).values
                / mc_actions.shape[1]
            )
        return estimated_policy

    def evaluate(
        self,
        mab: BaseMab,
        n_mc_experiments: int = 1000,
        save_path: Optional[str] = None,
        visualize: bool = True,
    ) -> pd.DataFrame:
        """
        Execute the OPE process with multiple estimators simultaneously.

        Parameters
        ----------
        mab : BaseMab
            Multi-armed bandit model to be evaluated
        n_mc_experiments : int
            Number of Monte Carlo experiments for policy estimation
        save_path : Optional[str], defaults to None.
            Path to save the results. Nothing is saved if not specified.
        visualize : bool, defaults to True.
            Whether to visualize the results of the OPE process

        Returns
        -------
        estimated_policy_value_df : pd.DataFrame
            Estimated policy values and confidence intervals
        """
        if visualize and not save_path and not in_jupyter_notebook():
            raise ValueError("save_path is required for visualization when not running in a Jupyter notebook")

        # Define OPE keyword arguments
        kwargs = {}
        if self._check_argument_required_by_estimators("action", self.ope_estimators):
            kwargs["action"] = self._test_data["action"]
        if self._check_argument_required_by_estimators("estimated_policy", self.ope_estimators):
            kwargs["estimated_policy"] = self._estimate_policy(mab=mab, n_mc_experiments=n_mc_experiments)
        if self._check_argument_required_by_estimators("propensity_score", self.ope_estimators):
            kwargs["propensity_score"] = self._test_data["propensity_score"]
        if self._check_argument_required_by_estimators("expected_importance_weight", self.ope_estimators):
            kwargs["expected_importance_weight"] = self._estimate_importance_weight(mab)

        # Instantiate class to conduct OPE by multiple estimators simultaneously
        multi_objective_estimated_policy_value_df = pd.DataFrame()
        results = {"value": [], "lower": [], "upper": [], "std": [], "estimator": [], "objective": []}
        for reward_feature in self.reward_feature:
            if self.verbose:
                logger.info(f"Offline Policy Evaluation for {reward_feature}.")

            if self._check_argument_required_by_estimators("reward", self.ope_estimators):
                kwargs["reward"] = self._test_data["reward"][:, self.reward_feature.index(reward_feature)]
            if self._check_argument_required_by_estimators("expected_reward", self.ope_estimators):
                kwargs["expected_reward"] = self._estimated_expected_reward[reward_feature]

            # Summarize policy values and their confidence intervals estimated by OPE estimators
            for ope_estimator in self.ope_estimators:
                estimated_policy_value, low, high, std = ope_estimator.estimate_policy_value_with_confidence_interval(
                    **kwargs,
                )
                results["value"].append(estimated_policy_value)
                results["lower"].append(low)
                results["upper"].append(high)
                results["std"].append(std)
                results["estimator"].append(ope_estimator.name)
                results["objective"].append(reward_feature)

            multi_objective_estimated_policy_value_df = pd.concat(
                [multi_objective_estimated_policy_value_df, pd.DataFrame.from_dict(results)],
                axis=0,
            )
        if save_path:
            multi_objective_estimated_policy_value_df.to_csv(os.path.join(save_path, "estimated_policy_value.csv"))

        if visualize:
            self._visualize_results(save_path, multi_objective_estimated_policy_value_df)

        return multi_objective_estimated_policy_value_df

    def update_and_evaluate(
        self,
        mab: BaseMab,
        n_mc_experiments: int = 1000,
        save_path: Optional[str] = None,
        visualize: bool = True,
        with_test: bool = False,
    ) -> pd.DataFrame:
        """
        Execute update of the multi-armed bandit based on the logged data,
        followed by the OPE process with multiple estimators simultaneously.

        Parameters
        ----------
        mab : BaseMab
            Multi-armed bandit model to be updated and evaluated
        n_mc_experiments : int
            Number of Monte Carlo experiments for policy estimation
        save_path : Optional[str]
            Path to save the results. Nothing is saved if not specified.
        visualize : bool
            Whether to visualize the results of the OPE process
        with_test : bool
            Whether to update the bandit model with the test data

        Returns
        -------
        estimated_policy_value_df : pd.DataFrame
            Estimated policy values
        """
        self._update_mab(mab, self._train_data)
        if with_test:
            self._update_mab(mab, self._test_data)
        estimated_policy_value_df = self.evaluate(mab, n_mc_experiments, save_path, visualize)
        return estimated_policy_value_df

    def _update_mab(self, mab: BaseMab, data: Dict[str, Any]):
        """
        Update the multi-armed bandit model based on the logged data.

        Parameters
        ----------
        mab : BaseMab
            Multi-armed bandit model to be updated.
        data : Dict[str, Any]
            Data used to update the bandit model.
        """
        if self.verbose:
            logger.info(f"Offline policy update for {type(mab)}.")
        kwargs = {"context": data["context"]} if self.contextual_features else {}
        mab.update(actions=data["action_ids"].tolist(), rewards=np.squeeze(data["reward"]).tolist(), **kwargs)

    def _visualize_results(self, save_path: Optional[str], multi_objective_estimated_policy_value_df: pd.DataFrame):
        """
        Visualize the results of the OPE process.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save the visualization results. Required if not running in a Jupyter notebook.
        multi_objective_estimated_policy_value_df : pd.DataFrame
            Estimated confidence intervals
        """

        tabs = []
        grouped_df = multi_objective_estimated_policy_value_df.groupby("objective")
        tools = "crosshair, pan, wheel_zoom, box_zoom, reset, hover, save"

        tooltips = [
            ("Estimator", "@estimator"),
            ("Estimated policy value", "@value"),
            ("Lower CI", "@lower"),
            ("Upper CI", "@upper"),
        ]
        for group_name, estimated_interval_df in grouped_df:
            source = ColumnDataSource(
                data=dict(
                    estimator=estimated_interval_df["estimator"],
                    value=estimated_interval_df["value"],
                    lower=estimated_interval_df["lower"],
                    upper=estimated_interval_df["upper"],
                )
            )
            fig = figure(
                title=f"Policy value estimates for {group_name} objective",
                x_axis_label="Estimator",
                y_axis_label="Estimated policy value (\u00b1 CI)",
                sizing_mode="inherit",
                x_range=source.data["estimator"],
                tools=tools,
                tooltips=tooltips,
            )
            fig.vbar(x="estimator", top="value", width=0.9, source=source)

            # Add error bars for confidence intervals
            fig.segment(
                x0="estimator", y0="lower", x1="estimator", y1="upper", source=source, line_width=2, color="black"
            )  # error bar line
            fig.vbar(
                x="estimator", width=0.1, bottom="lower", top="upper", source=source, color="black"
            )  # error bar cap

            fig.xgrid.grid_line_color = None

            tabs.append(TabPanel(child=fig, title=f"{group_name}"))

        output_path = os.path.join(save_path, "multi_objective_estimated_policy.html") if save_path else None
        visualize_via_bokeh(tabs=tabs, output_path=output_path)


def _mab_predict(mab: BaseMab, mab_data: Union[np.ndarray, PositiveInt], mc_experiment: int = 0) -> List[ActionId]:
    """
    bandit action probabilities prediction in test set

    Parameters
    ----------
    mab : BaseMab
        Multi-armed bandit model
    mab_data : Union[np.ndarray, PositiveInt]
        test data used to update the bandit model; context or number of samples.
    mc_experiment : int
        placeholder for multiprocessing

    Returns
    -------
    actions: List[ActionId] of shape (n_samples,)
        The actions selected by the multi-armed bandit model.
    """
    mab_output = mab.predict(context=mab_data) if type(mab_data) is np.ndarray else mab.predict(n_samples=mab_data)
    actions = mab_output[0]
    return actions
