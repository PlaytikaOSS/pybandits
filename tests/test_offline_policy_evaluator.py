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
import pydantic
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import PositiveInt
from pytest_mock import MockerFixture
from sklearn.preprocessing import MinMaxScaler

import pybandits
from pybandits import offline_policy_estimator
from pybandits.cmab import CmabBernoulli, CmabBernoulliCC, CmabBernoulliMO, CmabBernoulliMOCC
from pybandits.offline_policy_estimator import BaseOfflinePolicyEstimator
from pybandits.offline_policy_evaluator import OfflinePolicyEvaluator, _FunctionEstimator
from pybandits.pydantic_version_compatibility import (
    PYDANTIC_VERSION_1,
    PYDANTIC_VERSION_2,
    ValidationError,
    pydantic_version,
)
from pybandits.smab import (
    SmabBernoulli,
    SmabBernoulliCC,
    SmabBernoulliMO,
    SmabBernoulliMOCC,
)
from pybandits.utils import get_non_abstract_classes
from tests.utils import mock_update


@pytest.fixture(scope="module")
def logged_data(n_samples=10, n_actions=2, n_batches=3, n_rewards=2, n_groups=2, n_features=3):
    unique_actions = [f"a{i}" for i in range(n_actions)]
    action_ids = np.random.choice(unique_actions, n_samples * n_batches)
    batches = [i for i in range(n_batches) for _ in range(n_samples)]
    rewards = [np.random.randint(2, size=(n_samples * n_batches)) for _ in range(n_rewards)]
    action_true_rewards = {(a, r): np.random.rand() for a in unique_actions for r in range(n_rewards)}
    true_rewards = [
        np.array([action_true_rewards[(a, r)] for a in action_ids]).reshape(n_samples * n_batches)
        for r in range(n_rewards)
    ]
    groups = np.random.randint(n_groups, size=n_samples * n_batches)
    action_costs = {action: np.random.rand() for action in unique_actions}
    costs = np.array([action_costs[a] for a in action_ids])
    context = np.random.rand(n_samples * n_batches, n_features)
    action_propensity_score = {action: np.random.rand() for action in unique_actions}
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
        if pydantic_version == PYDANTIC_VERSION_1:
            pydantic.class_validators._FUNCS.clear()
            importlib.reload(pybandits.offline_policy_evaluator)
        elif pydantic_version == PYDANTIC_VERSION_2:
            importlib.reload(pybandits.offline_policy_evaluator)
        else:
            raise ValueError(f"Unsupported pydantic version: {pydantic_version}")

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


def test_safe_cv_raises_on_single_sample_class(labels=np.array([0, 0, 0, 0, 1])) -> None:
    """Test that _safe_cv raises ValueError when a class has fewer than 2 samples."""
    with pytest.raises(ValueError, match="insufficient for cross-validation"):
        _FunctionEstimator._safe_cv(labels)
