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
import importlib
import random
import sys
from tempfile import TemporaryDirectory
from typing import List, Optional, Union, get_args, get_type_hints
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import PositiveInt, ValidationError
from pytest_mock import MockerFixture
from sklearn.preprocessing import MinMaxScaler

import pybandits
from pybandits import offline_policy_estimator
from pybandits.cmab import CmabBernoulli, CmabBernoulliCC, CmabBernoulliMO, CmabBernoulliMOCC
from pybandits.offline_policy_estimator import BaseOfflinePolicyEstimator
from pybandits.offline_policy_evaluator import OfflinePolicyEvaluator, _FunctionEstimator, _mab_predict_serialized
from pybandits.smab import (
    SmabBernoulli,
    SmabBernoulliCC,
    SmabBernoulliMO,
    SmabBernoulliMOCC,
)
from pybandits.utils import get_non_abstract_classes
from tests.utils import mock_update


@pytest.fixture(scope="module")
def logged_data(rng, n_samples=10, n_actions=2, n_batches=3, n_rewards=2, n_groups=2, n_features=3):
    unique_actions = [f"a{i}" for i in range(n_actions)]
    action_ids = rng.choice(unique_actions, n_samples * n_batches)
    batches = [i for i in range(n_batches) for _ in range(n_samples)]
    rewards = [rng.integers(2, size=(n_samples * n_batches)) for _ in range(n_rewards)]
    action_true_rewards = {(a, r): rng.random() for a in unique_actions for r in range(n_rewards)}
    true_rewards = [
        np.array([action_true_rewards[(a, r)] for a in action_ids]).reshape(n_samples * n_batches)
        for r in range(n_rewards)
    ]
    groups = rng.integers(n_groups, size=n_samples * n_batches)
    action_costs = {action: rng.random() for action in unique_actions}
    costs = np.array([action_costs[a] for a in action_ids])
    context = rng.random((n_samples * n_batches, n_features))
    action_propensity_score = {action: rng.random() for action in unique_actions}
    propensity_score = np.array([action_propensity_score[a] for a in action_ids])
    return pd.DataFrame(
        {
            "batch": batches,
            "action_id": action_ids,
            "cost": costs,
            "group": groups,
            **{f"reward_{r}": rewards[r] for r in range(n_rewards)},
            **{f"true_reward_{r}": true_rewards[r] for r in range(n_rewards)},
            **{f"context_{i}": context[:, i] for i in range(n_features)},
            "propensity_score": propensity_score,
        }
    )


# validate failure for empty logged_data
def test_empty_logged_data(
    split_prop=0.5,
    n_trials=10,
    ope_estimators=None,
    verbose=False,
    batch_feature="batch",
    action_feature="action_id",
    reward_feature="reward",
    propensity_score_model_type="empirical",
    expected_reward_model_type="logreg",
    importance_weights_model_type="logreg",
):
    evaluator = OfflinePolicyEvaluator(
        split_prop=split_prop,
        propensity_score_model_type=propensity_score_model_type,
        expected_reward_model_type=expected_reward_model_type,
        importance_weights_model_type=importance_weights_model_type,
        n_trials=n_trials,
        ope_estimators=ope_estimators,
        batch_feature=batch_feature,
        action_feature=action_feature,
        reward_feature=reward_feature,
        verbose=verbose,
    )
    with pytest.raises(AttributeError):
        evaluator._validate_logged_data(pd.DataFrame())


@pytest.mark.usefixtures("logged_data")
@given(
    split_prop=st.sampled_from([0.0, 1.0]),
    n_trials=st.just(10),
    ope_estimators=st.just(None),
    verbose=st.just(False),
    batch_feature=st.just("batch"),
    action_feature=st.just("action_id"),
    reward_feature=st.just("reward_0"),
    propensity_score_model_type=st.just("empirical"),
    expected_reward_model_type=st.just("logreg"),
    importance_weights_model_type=st.just("logreg"),
)
# validate failure for extreme split_prop values
def test_initialization_extreme_split_prop(
    logged_data: MockerFixture,
    split_prop: float,
    n_trials: PositiveInt,
    ope_estimators: Optional[List[BaseOfflinePolicyEstimator]],
    verbose: bool,
    batch_feature: str,
    action_feature: str,
    reward_feature: str,
    propensity_score_model_type: str,
    expected_reward_model_type: str,
    importance_weights_model_type: str,
):
    with pytest.raises(ValueError):
        OfflinePolicyEvaluator(
            split_prop=split_prop,
            propensity_score_model_type=propensity_score_model_type,
            expected_reward_model_type=expected_reward_model_type,
            importance_weights_model_type=importance_weights_model_type,
            n_trials=n_trials,
            ope_estimators=ope_estimators,
            batch_feature=batch_feature,
            action_feature=action_feature,
            reward_feature=reward_feature,
            true_reward_feature=reward_feature,
            verbose=verbose,
        )


# validate failure for invalid initialization parameters
def test_initialization_mismatches(
    logged_data: MockerFixture,
    split_prop=0.5,
    n_trials=10,
    ope_estimators=None,
    verbose=False,
    batch_feature="batch",
    action_feature="action_id",
    reward_feature="reward_0",
    propensity_score_model_type="empirical",
    expected_reward_model_type="logreg",
    importance_weights_model_type="logreg",
):
    # more true_reward_features than rewards
    with pytest.raises(ValueError):
        OfflinePolicyEvaluator(
            split_prop=split_prop,
            propensity_score_model_type=propensity_score_model_type,
            expected_reward_model_type=expected_reward_model_type,
            importance_weights_model_type=importance_weights_model_type,
            n_trials=n_trials,
            ope_estimators=ope_estimators,
            batch_feature=batch_feature,
            action_feature=action_feature,
            reward_feature=reward_feature,
            true_reward_feature=[reward_feature, reward_feature],
            verbose=verbose,
        )
    # missing propensity_score_feature
    with pytest.raises(ValueError):
        OfflinePolicyEvaluator(
            split_prop=split_prop,
            propensity_score_model_type="propensity_score",
            expected_reward_model_type=expected_reward_model_type,
            importance_weights_model_type=importance_weights_model_type,
            n_trials=n_trials,
            ope_estimators=ope_estimators,
            batch_feature=batch_feature,
            action_feature=action_feature,
            reward_feature=reward_feature,
        )
    # missing context
    evaluator_with_bad_context = OfflinePolicyEvaluator(
        split_prop=split_prop,
        propensity_score_model_type=propensity_score_model_type,
        expected_reward_model_type=expected_reward_model_type,
        importance_weights_model_type=importance_weights_model_type,
        n_trials=n_trials,
        ope_estimators=ope_estimators,
        batch_feature=batch_feature,
        action_feature=action_feature,
        reward_feature=reward_feature,
        verbose=False,
        contextual_features=["non_existent"],
    )
    with pytest.raises(AttributeError):
        evaluator_with_bad_context._validate_logged_data(logged_data)


def generate_random_bool() -> bool:
    return np.random.rand() > 0.5


@pytest.mark.usefixtures("logged_data")
@settings(deadline=None, max_examples=20)
@given(
    split_prop=st.just(0.5),
    n_trials=st.just(2),
    propensity_score_model_type=st.sampled_from(
        get_args(get_type_hints(OfflinePolicyEvaluator)["propensity_score_model_type"])
    ),
    expected_reward_model_type=st.sampled_from(
        get_args(get_type_hints(OfflinePolicyEvaluator)["expected_reward_model_type"])
    ),
    importance_weights_model_type=st.sampled_from(
        get_args(get_type_hints(OfflinePolicyEvaluator)["importance_weights_model_type"])
    ),
    batch_feature=st.just("batch"),
    action_feature=st.just("action_id"),
    reward_feature=st.sampled_from(["reward_0", ["reward_0", "reward_1"]]),
    context=st.booleans(),
    group_feature=st.sampled_from(["group", None]),
    cost_feature=st.sampled_from(["cost", None]),
    propensity_score_feature=st.just("propensity_score"),
    n_mc_experiments=st.just(2),
)
# test various OfflinePolicyEvaluator configurations to validate that everything works
def test_running_configuration(
    logged_data: MockerFixture,
    split_prop: float,
    n_trials: PositiveInt,
    propensity_score_model_type: str,
    expected_reward_model_type: str,
    importance_weights_model_type: str,
    batch_feature: str,
    action_feature: str,
    reward_feature: Union[str, List[int]],
    context: bool,
    group_feature: Optional[str],
    cost_feature: Optional[str],
    propensity_score_feature: Optional[str],
    n_mc_experiments: int,
    monkeymodule,
):
    ope_estimators = random.choice(get_non_abstract_classes(offline_policy_estimator) + [None])
    scaler = random.choice([None, MinMaxScaler()])
    if ope_estimators is not None:
        ope_estimators = [ope_estimators()]
    shuffle = generate_random_bool()
    update = generate_random_bool()
    visualize = generate_random_bool()
    verbose = generate_random_bool()
    fast_fit = generate_random_bool()

    true_reward_feature = (
        f"true_{reward_feature}" if isinstance(reward_feature, str) else [f"true_{r}" for r in reward_feature]
    )
    if context:
        contextual_features = [col for col in logged_data.columns if col.startswith("context")]
        monkeymodule.setattr(
            pybandits.model.BaseBayesianNeuralNetwork,
            "_update",
            mock_update,
        )

    else:
        contextual_features = None
    unique_actions = logged_data["action_id"].unique()
    if cost_feature:
        action_ids_cost = {
            action_id: logged_data["cost"][logged_data["action_id"] == action_id].iloc[0]
            for action_id in unique_actions
        }
    if context:
        if cost_feature:
            if type(reward_feature) is list:
                mab = CmabBernoulliMOCC.cold_start(
                    action_ids_cost=action_ids_cost,
                    n_objectives=len(reward_feature),
                    n_features=len(contextual_features),
                )
            else:
                mab = CmabBernoulliCC.cold_start(action_ids_cost=action_ids_cost, n_features=len(contextual_features))
        else:
            if type(reward_feature) is list:
                mab = CmabBernoulliMO.cold_start(
                    action_ids=set(unique_actions),
                    n_objectives=len(reward_feature),
                    n_features=len(contextual_features),
                )
            else:
                mab = CmabBernoulli.cold_start(action_ids=set(unique_actions), n_features=len(contextual_features))
    else:
        if cost_feature:
            if type(reward_feature) is list:
                mab = SmabBernoulliMOCC.cold_start(action_ids_cost=action_ids_cost, n_objectives=len(reward_feature))
            else:
                mab = SmabBernoulliCC.cold_start(action_ids_cost=action_ids_cost)
        else:
            if type(reward_feature) is list:
                mab = SmabBernoulliMO.cold_start(action_ids=set(unique_actions), n_objectives=len(reward_feature))
            else:
                mab = SmabBernoulli.cold_start(action_ids=set(unique_actions))
    evaluator = OfflinePolicyEvaluator(
        split_prop=split_prop,
        n_trials=n_trials,
        fast_fit=fast_fit,
        scaler=scaler,
        ope_estimators=ope_estimators,
        verbose=verbose,
        shuffle=shuffle,
        propensity_score_model_type=propensity_score_model_type,
        expected_reward_model_type=expected_reward_model_type,
        importance_weights_model_type=importance_weights_model_type,
        batch_feature=batch_feature,
        action_feature=action_feature,
        reward_feature=reward_feature,
        true_reward_feature=true_reward_feature,
        contextual_features=contextual_features,
        group_feature=group_feature,
        cost_feature=cost_feature,
        propensity_score_feature=propensity_score_feature,
    )
    execution_func = evaluator.update_and_evaluate if update else evaluator.evaluate
    with TemporaryDirectory() as tmp_dir:
        execution_func(
            mab=mab, logged_data=logged_data, visualize=visualize, n_mc_experiments=n_mc_experiments, save_path=tmp_dir
        )


@pytest.mark.usefixtures("logged_data")
@settings(deadline=None)
@given(
    split_prop=st.just(0.5),
    n_trials=st.just(2),
    propensity_score_model_type=st.sampled_from(
        get_args(get_type_hints(OfflinePolicyEvaluator)["propensity_score_model_type"])
    ),
    expected_reward_model_type=st.sampled_from(
        get_args(get_type_hints(OfflinePolicyEvaluator)["expected_reward_model_type"])
    ),
    importance_weights_model_type=st.sampled_from(
        get_args(get_type_hints(OfflinePolicyEvaluator)["importance_weights_model_type"])
    ),
    batch_feature=st.just("batch"),
    action_feature=st.just("action_id"),
    reward_feature=st.sampled_from(["reward_0", ["reward_0", "reward_1"]]),
    context=st.booleans(),
    group_feature=st.sampled_from(["group", None]),
    cost_feature=st.sampled_from(["cost", None]),
    propensity_score_feature=st.just("propensity_score"),
    ope_estimators=st.just(None),
)
def test_initialization_when_xgboost_not_available(
    logged_data: pd.DataFrame,
    split_prop: float,
    n_trials: PositiveInt,
    propensity_score_model_type: str,
    expected_reward_model_type: str,
    importance_weights_model_type: str,
    batch_feature: str,
    action_feature: str,
    reward_feature: Union[str, List[str]],
    context: bool,
    group_feature: Optional[str],
    cost_feature: Optional[str],
    propensity_score_feature: Optional[str],
    ope_estimators: Optional[List[BaseOfflinePolicyEstimator]],
    monkeymodule,
) -> None:
    """Test that other model types still work when XGBoost is not available."""
    with patch.dict(sys.modules, {"xgboost": None}):
        importlib.reload(pybandits.offline_policy_evaluator)

        assert pybandits.offline_policy_evaluator._XGBOOST_AVAILABLE is False
        assert pybandits.offline_policy_evaluator.XGBClassifier is None
        OfflinePolicyEvaluator = importlib.import_module("pybandits.offline_policy_evaluator").OfflinePolicyEvaluator

        scaler = random.choice([None, MinMaxScaler()])
        shuffle = generate_random_bool()
        verbose = generate_random_bool()
        fast_fit = generate_random_bool()

        true_reward_feature = (
            f"true_{reward_feature}" if isinstance(reward_feature, str) else [f"true_{r}" for r in reward_feature]
        )
        if context:
            contextual_features = [col for col in logged_data.columns if col.startswith("context")]
            monkeymodule.setattr(
                pybandits.model.BaseBayesianNeuralNetwork,
                "_update",
                mock_update,
            )
        else:
            contextual_features = None

        if "xgb" in (propensity_score_model_type, expected_reward_model_type, importance_weights_model_type):
            # When XGBoost is not available, using "xgb" should fail at runtime
            with pytest.raises(ValidationError):
                OfflinePolicyEvaluator(
                    split_prop=split_prop,
                    propensity_score_model_type=propensity_score_model_type,
                    expected_reward_model_type=expected_reward_model_type,
                    importance_weights_model_type=importance_weights_model_type,
                    n_trials=n_trials,
                    fast_fit=fast_fit,
                    scaler=scaler,
                    ope_estimators=ope_estimators,
                    verbose=verbose,
                    shuffle=shuffle,
                    batch_feature=batch_feature,
                    action_feature=action_feature,
                    reward_feature=reward_feature,
                    contextual_features=[col for col in logged_data.columns if col.startswith("context")]
                    if context
                    else None,
                    group_feature=group_feature,
                    cost_feature=cost_feature,
                    propensity_score_feature=propensity_score_feature,
                )
        else:
            evaluator = OfflinePolicyEvaluator(
                split_prop=split_prop,
                propensity_score_model_type=propensity_score_model_type,
                expected_reward_model_type=expected_reward_model_type,
                importance_weights_model_type=importance_weights_model_type,
                n_trials=n_trials,
                fast_fit=fast_fit,
                scaler=scaler,
                ope_estimators=ope_estimators,
                verbose=verbose,
                shuffle=shuffle,
                batch_feature=batch_feature,
                action_feature=action_feature,
                reward_feature=reward_feature,
                true_reward_feature=true_reward_feature,
                contextual_features=contextual_features,
                group_feature=group_feature,
                cost_feature=cost_feature,
                propensity_score_feature=propensity_score_feature,
            )
            assert evaluator is not None


def test_safe_cv_raises_on_single_sample_class(labels=(0, 0, 0, 0, 1)) -> None:
    """Test that _safe_cv raises ValueError when a class has fewer than 2 samples."""
    with pytest.raises(ValueError, match="insufficient for cross-validation"):
        _FunctionEstimator._safe_cv(np.array(labels))


# ──────────────────────────────────────────────────────
# Additional tests: grouped by class
# ──────────────────────────────────────────────────────


class TestFunctionEstimatorEdgeCases:
    """Edge-case tests for _FunctionEstimator."""

    @pytest.fixture(scope="module")
    def base_estimator_kwargs(self) -> dict:
        """Minimal construction kwargs for a non-multiclass _FunctionEstimator."""
        return {
            "estimator_type": "logreg",
            "fast_fit": True,
            "n_trials": 2,
            "verbose": False,
            "multi_action_prediction": False,
        }

    @pytest.mark.parametrize(
        "estimator_type",
        get_args(get_type_hints(_FunctionEstimator)["estimator_type"]),
    )
    def test_include_action_false_multi_action_true_raises(
        self, estimator_type: str, base_estimator_kwargs: dict
    ) -> None:
        """include_action_in_features=False with multi_action_prediction=True raises ValidationError."""
        with pytest.raises(ValidationError):
            _FunctionEstimator(
                **{
                    **base_estimator_kwargs,
                    "estimator_type": estimator_type,
                    "multi_action_prediction": True,
                    "include_action_in_features": False,
                },
            )

    @given(n_features=st.integers(min_value=1, max_value=5))
    @settings(max_examples=5, deadline=None)
    def test_predict_before_fit_raises(self, n_features: int, base_estimator_kwargs: dict) -> None:
        """Calling predict before fit raises AttributeError for any context shape."""
        estimator = _FunctionEstimator(**base_estimator_kwargs)
        n_rounds = n_features
        with pytest.raises(AttributeError):
            estimator.predict(
                {
                    "context": np.zeros((n_rounds, n_features)),
                    "action_ids": np.array([f"a{i}" for i in range(n_rounds)]),
                    "n_rounds": n_rounds,
                    "unique_actions": [f"a{i}" for i in range(n_rounds)],
                }
            )

    @given(
        n_samples=st.integers(min_value=10, max_value=30),
        n_features=st.integers(min_value=2, max_value=5),
        n_actions=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=5, deadline=None)
    def test_predict_multiclass_path(
        self, n_samples: int, n_features: int, n_actions: int, base_estimator_kwargs: dict, rng: np.random.Generator
    ) -> None:
        """include_action_in_features=False exercises the multiclass probability extraction path."""
        n_test = min(n_actions, n_samples)
        unique_labels = [f"a{i}" for i in range(n_actions)]
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        actions = np.array([f"a{i % n_actions}" for i in range(n_samples)])
        encoded = np.array([label_to_int[a] for a in actions])
        context = rng.random((n_samples, n_features))

        estimator = _FunctionEstimator(
            **{**base_estimator_kwargs, "include_action_in_features": False, "calibrate": False},
        )
        estimator.fit(
            X={"context": context, "action_ids": actions, "action": encoded, "n_rounds": n_samples},
            y=encoded,
        )
        prediction = estimator.predict(
            {
                "context": context[:n_test],
                "action_ids": actions[:n_test],
                "action": encoded[:n_test],
                "n_rounds": n_test,
                "unique_actions": unique_labels,
            }
        )
        assert prediction.shape == (n_test,)
        assert np.all((prediction >= 0) & (prediction <= 1))


class TestValidateLoggedData:
    """Tests for OfflinePolicyEvaluator._validate_logged_data error paths."""

    @pytest.fixture
    def feature_names(self) -> dict:
        """Column names used consistently by the evaluator and the DataFrame fixtures."""
        return {"batch": "batch", "action": "action_id", "reward": "reward"}

    @pytest.fixture
    def base_evaluator(self, feature_names: dict) -> OfflinePolicyEvaluator:
        """Standard evaluator configured with feature_names columns."""
        return OfflinePolicyEvaluator(
            split_prop=0.5,
            propensity_score_model_type="empirical",
            expected_reward_model_type="logreg",
            importance_weights_model_type="logreg",
            ope_estimators=None,
            n_trials=2,
            batch_feature=feature_names["batch"],
            action_feature=feature_names["action"],
            reward_feature=feature_names["reward"],
        )

    @pytest.fixture
    def evaluator_with_true_reward(self, feature_names: dict) -> OfflinePolicyEvaluator:
        """Evaluator that additionally expects a true_reward column."""
        return OfflinePolicyEvaluator(
            split_prop=0.5,
            propensity_score_model_type="empirical",
            expected_reward_model_type="logreg",
            importance_weights_model_type="logreg",
            ope_estimators=None,
            n_trials=2,
            batch_feature=feature_names["batch"],
            action_feature=feature_names["action"],
            reward_feature=feature_names["reward"],
            true_reward_feature="true_reward",
        )

    @pytest.fixture
    def evaluator_with_cost(self, feature_names: dict) -> OfflinePolicyEvaluator:
        """Evaluator that additionally expects a cost_col column."""
        return OfflinePolicyEvaluator(
            split_prop=0.5,
            propensity_score_model_type="empirical",
            expected_reward_model_type="logreg",
            importance_weights_model_type="logreg",
            ope_estimators=None,
            n_trials=2,
            batch_feature=feature_names["batch"],
            action_feature=feature_names["action"],
            reward_feature=feature_names["reward"],
            cost_feature="cost_col",
        )

    @pytest.fixture
    def batch_wrong_type_df(self, feature_names: dict) -> pd.DataFrame:
        """DataFrame whose batch column contains strings instead of ints."""
        return pd.DataFrame(
            {
                feature_names["batch"]: ["x", "y"],
                feature_names["action"]: ["a", "b"],
                feature_names["reward"]: [1, 0],
            }
        )

    @pytest.fixture
    def missing_action_df(self, feature_names: dict) -> pd.DataFrame:
        """DataFrame that is missing the action column."""
        return pd.DataFrame(
            {
                feature_names["batch"]: [0, 1],
                feature_names["reward"]: [1, 0],
            }
        )

    @pytest.fixture
    def missing_reward_df(self, feature_names: dict) -> pd.DataFrame:
        """DataFrame that is missing the reward column."""
        return pd.DataFrame(
            {
                feature_names["batch"]: [0, 1],
                feature_names["action"]: ["a", "b"],
            }
        )

    @pytest.fixture
    def valid_base_df(self, feature_names: dict) -> pd.DataFrame:
        """Minimal valid DataFrame that satisfies the base evaluator but lacks optional columns."""
        return pd.DataFrame(
            {
                feature_names["batch"]: [0, 1],
                feature_names["action"]: ["a", "b"],
                feature_names["reward"]: [1, 0],
            }
        )

    @pytest.mark.parametrize(
        "evaluator_fixture, data_fixture, expected_exc",
        [
            ("base_evaluator", "batch_wrong_type_df", TypeError),
            ("base_evaluator", "missing_action_df", AttributeError),
            ("base_evaluator", "missing_reward_df", AttributeError),
            ("evaluator_with_true_reward", "valid_base_df", AttributeError),
            ("evaluator_with_cost", "valid_base_df", AttributeError),
        ],
    )
    def test_validate_raises(
        self,
        evaluator_fixture: str,
        data_fixture: str,
        expected_exc: type,
        request: pytest.FixtureRequest,
    ) -> None:
        """_validate_logged_data raises the expected exception for each invalid input combination."""
        evaluator = request.getfixturevalue(evaluator_fixture)
        df = request.getfixturevalue(data_fixture)
        with pytest.raises(expected_exc):
            evaluator._validate_logged_data(df)


class TestOfflinePolicyEvaluatorPipeline:
    """Tests for OfflinePolicyEvaluator pipeline paths not covered by the integration tests."""

    @pytest.fixture(scope="module")
    def contextual_features(self, logged_data: pd.DataFrame) -> List[str]:
        """All context_* column names present in logged_data."""
        return [col for col in logged_data.columns if col.startswith("context_")]

    @pytest.fixture(scope="module")
    def first_reward_feature(self, logged_data: pd.DataFrame) -> str:
        """The first reward_* column name in logged_data."""
        return next(col for col in logged_data.columns if col.startswith("reward_"))

    @pytest.fixture(scope="module")
    def base_evaluator(self, logged_data: pd.DataFrame, first_reward_feature: str) -> OfflinePolicyEvaluator:
        """Standard fast-fit evaluator using empirical propensity and logreg expected-reward."""
        return OfflinePolicyEvaluator(
            split_prop=0.5,
            propensity_score_model_type="empirical",
            expected_reward_model_type="logreg",
            importance_weights_model_type="logreg",
            ope_estimators=None,
            n_trials=2,
            fast_fit=True,
            batch_feature="batch",
            action_feature="action_id",
            reward_feature=first_reward_feature,
        )

    @pytest.fixture(scope="module")
    def dict_scaler_evaluator(
        self, logged_data: pd.DataFrame, contextual_features: List[str], first_reward_feature: str
    ) -> OfflinePolicyEvaluator:
        """Evaluator configured with a per-feature dict scaler over contextual_features."""
        return OfflinePolicyEvaluator(
            split_prop=0.5,
            propensity_score_model_type="empirical",
            expected_reward_model_type="logreg",
            importance_weights_model_type="logreg",
            ope_estimators=None,
            n_trials=2,
            fast_fit=True,
            batch_feature="batch",
            action_feature="action_id",
            reward_feature=first_reward_feature,
            contextual_features=contextual_features,
            scaler={feature: MinMaxScaler() for feature in contextual_features},
        )

    def test_extract_batches_with_dict_scaler(
        self,
        logged_data: pd.DataFrame,
        dict_scaler_evaluator: OfflinePolicyEvaluator,
        contextual_features: List[str],
    ) -> None:
        """_extract_batches applies a per-feature dict scaler and produces correct context shape."""
        train_data, test_data = dict_scaler_evaluator._extract_batches(logged_data)
        assert train_data["context"].shape[1] == len(contextual_features)
        assert test_data["context"].shape[1] == len(contextual_features)

    @given(n_mc=st.integers(min_value=1, max_value=3))
    @settings(max_examples=3, deadline=None)
    def test_estimate_policy_no_cores(
        self, n_mc: int, logged_data: pd.DataFrame, base_evaluator: OfflinePolicyEvaluator
    ) -> None:
        """estimate_policy with n_cores=0 exercises the sequential (non-multiprocessing) path."""
        unique_actions = set(logged_data["action_id"].unique())
        mab = SmabBernoulli.cold_start(action_ids=unique_actions)
        _, test_data = base_evaluator._extract_batches(logged_data)
        estimated_policy = base_evaluator.estimate_policy(
            mab=mab, test_data=test_data, n_mc_experiments=n_mc, n_cores=0
        )
        assert isinstance(estimated_policy, pd.DataFrame)
        assert set(estimated_policy.columns) == unique_actions

    @given(n_mc=st.integers(min_value=1, max_value=3))
    @settings(max_examples=3, deadline=None)
    def test_update_and_evaluate_with_test(
        self, n_mc: int, logged_data: pd.DataFrame, base_evaluator: OfflinePolicyEvaluator
    ) -> None:
        """update_and_evaluate with with_test=True also updates the MAB on test-split data."""
        unique_actions = set(logged_data["action_id"].unique())
        mab = SmabBernoulli.cold_start(action_ids=unique_actions)
        with TemporaryDirectory() as tmp_dir:
            result = base_evaluator.update_and_evaluate(
                mab=mab,
                logged_data=logged_data,
                with_test=True,
                visualize=False,
                n_mc_experiments=n_mc,
                save_path=tmp_dir,
            )
        assert isinstance(result, pd.DataFrame)


class TestMabPredictSerializedUtils:
    """Tests for the module-level _mab_predict_serialized helper."""

    @pytest.fixture(scope="module")
    def serialized_smab(self, logged_data: pd.DataFrame) -> tuple:
        """(class_name, state) for a fresh SmabBernoulli built from logged_data's action set."""
        unique_actions = set(logged_data["action_id"].unique())
        mab = SmabBernoulli.cold_start(action_ids=unique_actions)
        return mab.get_state()

    @given(suffix=st.text(alphabet=st.characters(whitelist_categories=["Lu", "Ll"]), min_size=3, max_size=10))
    @settings(max_examples=5, deadline=None)
    def test_class_not_found_raises(self, suffix: str) -> None:
        """_mab_predict_serialized raises ValueError for any class name absent from pybandits modules."""
        invalid_class_name = f"_TestNonExistent_{suffix}_"
        with pytest.raises(ValueError, match="Could not find MAB class"):
            _mab_predict_serialized(
                mab_class_name=invalid_class_name,
                mab_state="{}",
                mab_data=2,
                verbose=False,
            )

    def test_import_error_is_handled(self) -> None:
        """ImportError during module import is caught and the function raises ValueError."""
        with patch("importlib.import_module", side_effect=ImportError("mocked")):
            with pytest.raises(ValueError, match="Could not find MAB class"):
                _mab_predict_serialized(
                    mab_class_name="SmabBernoulli",
                    mab_state="{}",
                    mab_data=2,
                    verbose=False,
                )

    @given(n_samples=st.integers(min_value=1, max_value=20))
    @settings(max_examples=5, deadline=None)
    def test_verbose_logging(self, n_samples: int, serialized_smab: tuple) -> None:
        """_mab_predict_serialized returns exactly n_samples actions when verbose=True."""
        mab_class_name, mab_state = serialized_smab
        actions = _mab_predict_serialized(
            mab_class_name=mab_class_name,
            mab_state=mab_state,
            mab_data=n_samples,
            verbose=True,
        )
        assert len(actions) == n_samples
