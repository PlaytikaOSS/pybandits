from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union, get_args, get_type_hints

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from matplotlib.pyplot import close
from pydantic import PositiveInt
from pytest_mock import MockerFixture
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from pybandits.cmab import CmabBernoulli, CmabBernoulliCC
from pybandits.offline_policy_estimator import BaseOfflinePolicyEstimator
from pybandits.offline_policy_evaluator import OfflinePolicyEvaluator
from pybandits.smab import (
    SmabBernoulli,
    SmabBernoulliCC,
    SmabBernoulliMO,
    SmabBernoulliMOCC,
)


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
    verbose=False,
    batch_feature="batch",
    action_feature="action_id",
    reward_feature="reward",
    propensity_score_model_type="empirical",
    expected_reward_model_type="logreg",
    importance_weights_model_type="logreg",
):
    with pytest.raises(AttributeError):
        OfflinePolicyEvaluator(
            logged_data=pd.DataFrame(),
            split_prop=split_prop,
            propensity_score_model_type=propensity_score_model_type,
            expected_reward_model_type=expected_reward_model_type,
            importance_weights_model_type=importance_weights_model_type,
            n_trials=n_trials,
            ope_metrics=None,
            batch_feature=batch_feature,
            action_feature=action_feature,
            reward_feature=reward_feature,
            verbose=verbose,
        )


@pytest.mark.usefixtures("logged_data")
@given(
    split_prop=st.sampled_from([0.0, 1.0]),
    n_trials=st.just(10),
    ope_metrics=st.just(None),
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
    ope_metrics: Optional[List[BaseOfflinePolicyEstimator]],
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
            logged_data=logged_data,
            split_prop=split_prop,
            propensity_score_model_type=propensity_score_model_type,
            expected_reward_model_type=expected_reward_model_type,
            importance_weights_model_type=importance_weights_model_type,
            n_trials=n_trials,
            ope_metrics=ope_metrics,
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
    ope_metrics=None,
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
            logged_data=logged_data,
            split_prop=split_prop,
            propensity_score_model_type=propensity_score_model_type,
            expected_reward_model_type=expected_reward_model_type,
            importance_weights_model_type=importance_weights_model_type,
            n_trials=n_trials,
            ope_metrics=ope_metrics,
            batch_feature=batch_feature,
            action_feature=action_feature,
            reward_feature=reward_feature,
            true_reward_feature=[reward_feature, reward_feature],
            verbose=verbose,
        )
    # missing propensity_score_feature
    with pytest.raises(ValueError):
        OfflinePolicyEvaluator(
            logged_data=logged_data,
            split_prop=split_prop,
            propensity_score_model_type="propensity_score",
            expected_reward_model_type=expected_reward_model_type,
            importance_weights_model_type=importance_weights_model_type,
            n_trials=n_trials,
            ope_metrics=ope_metrics,
            batch_feature=batch_feature,
            action_feature=action_feature,
            reward_feature=reward_feature,
            visualize=False,
        )
    # missing context
    with pytest.raises(AttributeError):
        OfflinePolicyEvaluator(
            logged_data=logged_data,
            split_prop=split_prop,
            propensity_score_model_type=propensity_score_model_type,
            expected_reward_model_type=expected_reward_model_type,
            importance_weights_model_type=importance_weights_model_type,
            n_trials=n_trials,
            ope_metrics=ope_metrics,
            batch_feature=batch_feature,
            action_feature=action_feature,
            reward_feature=reward_feature,
            verbose=False,
            contextual_features=["non_existent"],
        )


@pytest.mark.usefixtures("logged_data")
@settings(deadline=None)
@given(
    split_prop=st.just(0.5),
    n_trials=st.just(10),
    fast_fit=st.booleans(),
    scaler=st.sampled_from([None, MinMaxScaler()]),
    verbose=st.booleans(),
    visualize=st.booleans(),
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
    update=st.booleans(),
)
# test various OfflinePolicyEvaluator configurations to validate that everything works
def test_running_configuration(
    logged_data: MockerFixture,
    split_prop: float,
    n_trials: PositiveInt,
    fast_fit: bool,
    scaler: Optional[Union[TransformerMixin, Dict[str, TransformerMixin]]],
    verbose: bool,
    visualize: bool,
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
    update: bool,
):
    if context and type(reward_feature) is List:
        pass  # CmabMO and CmabMOCC are not supported yet
    true_reward_feature = (
        f"true_{reward_feature}" if isinstance(reward_feature, str) else [f"true_{r}" for r in reward_feature]
    )
    contextual_features = [col for col in logged_data.columns if col.startswith("context")] if context else None
    unique_actions = logged_data["action_id"].unique()
    if cost_feature:
        action_ids_cost = {
            action_id: logged_data["cost"][logged_data["action_id"] == action_id].iloc[0]
            for action_id in unique_actions
        }
    if context:
        if cost_feature:
            if type(reward_feature) is list:
                return  # CmabMOCC is not supported yet
            else:
                mab = CmabBernoulliCC.cold_start(action_ids_cost=action_ids_cost, n_features=len(contextual_features))
        else:
            if type(reward_feature) is list:
                return  # CmabMO is not supported yet
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
        logged_data=logged_data,
        split_prop=split_prop,
        n_trials=n_trials,
        fast_fit=fast_fit,
        scaler=scaler,
        ope_estimators=None,
        verbose=verbose,
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
        execution_func(mab=mab, visualize=visualize, n_mc_experiments=n_mc_experiments, save_path=tmp_dir)
    if visualize:
        close("all")  # close all figures to avoid memory leak
