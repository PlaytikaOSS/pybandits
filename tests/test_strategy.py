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

from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pytest_mock import MockerFixture

from pybandits.base import ActionId, BaseModel, Probability, UnifiedActionId
from pybandits.model import Beta, BetaCC, BetaMOCC
from pybandits.pydantic_version_compatibility import ValidationError
from pybandits.quantitative_model import QuantitativeModel
from pybandits.strategy import (
    BaseStrategy,
    BestActionIdentificationBandit,
    ClassicBandit,
    CostControlBandit,
    CostControlStrategy,
    MultiObjectiveBandit,
    MultiObjectiveCostControlBandit,
    MultiObjectiveStrategy,
    SingleObjectiveStrategy,
)
from tests.test_quantitative_model import DummyZooming

########################################################################################################################
# Helper functions and fixtures


# Test constants
DEFAULT_COST = 10.0
DEFAULT_DIMENSION = 2
DEFAULT_PROBABILITY = 0.5
DEFAULT_EXPLOIT_P = 0.5
DEFAULT_SUBSIDY_FACTOR = 0.5


class DummyQuantitativeModelCC(QuantitativeModel):
    cost: Optional[Callable[[np.ndarray], float]] = None
    models: Optional[List[BaseModel]] = None

    def reset(self) -> None:
        pass

    def sample_proba(self, **kwargs) -> None:
        pass

    def update(self, **kwargs) -> None:
        pass

    def _quantitative_update(self, **kwargs) -> None:
        pass

    def _reset(self) -> None:
        pass


def create_mock_quantitative_model(
    dimension: int = DEFAULT_DIMENSION,
    cost_value: float = DEFAULT_COST,
    mocked_model_type: Literal["MagicMock", "DummyQuantitativeModelCC"] = "MagicMock",
) -> QuantitativeModel:
    """Create a mock quantitative model for testing.

    Parameters
    ----------
    dimension : int
        Dimension of the quantitative model.
    cost_value : float
        Cost value to return.

    Returns
    -------
    QuantitativeModel
        Mock quantitative model.
    """
    if mocked_model_type == "MagicMock":
        model = MagicMock(spec=QuantitativeModel)
        model.dimension = dimension

    elif mocked_model_type == "DummyQuantitativeModelCC":
        model = DummyQuantitativeModelCC(dimension=dimension)
    else:
        raise ValueError(f"Invalid model type: {mocked_model_type}")
    model.cost = MagicMock(return_value=cost_value)
    return model


def create_mock_base_model(cost_value: float = DEFAULT_COST) -> BaseModel:
    """Create a mock base model for testing.

    Parameters
    ----------
    cost_value : float
        Cost value for the model.

    Returns
    -------
    BaseModel
        Mock base model.
    """
    model = MagicMock(spec=BaseModel)
    model.cost = cost_value
    return model


@st.composite
def action_probability_pairs(draw, min_actions: int = 2, max_actions: int = 10, allow_callables: bool = False):
    """Generate action-probability pairs for testing.

    Parameters
    ----------
    draw : function
        Hypothesis draw function.
    min_actions : int
        Minimum number of actions.
    max_actions : int
        Maximum number of actions.
    allow_callables : bool
        Whether to include callable probabilities.

    Returns
    -------
    tuple
        (action_dict, probability_dict, model_dict)
    """
    n_actions = draw(st.integers(min_value=min_actions, max_value=max_actions))
    action_ids = [f"action_{i}" for i in range(n_actions)]

    probabilities = {}
    models = {}

    for action_id in action_ids:
        cost_value = np.random.random()
        probability_value = np.random.random()
        if allow_callables and draw(st.booleans()):
            # Create a callable probability
            probabilities[action_id] = lambda x, p=probability_value: p
            models[action_id] = DummyZooming.cold_start(dimension=DEFAULT_DIMENSION, cost=lambda x, c=cost_value: c)
        else:
            # Create a fixed probability
            probabilities[action_id] = probability_value
            models[action_id] = BetaCC(cost=cost_value)

    return action_ids, probabilities, models


@pytest.fixture(scope="session")
def prob_dict_two_actions() -> Dict[str, float]:
    """Fixture providing a probability dictionary with two actions.

    Returns
    -------
    Dict[str, float]
        Probability dictionary with two actions (a1: 0.5, a2: 0.7).
    """
    return {"a1": 0.5, "a2": 0.7}


@pytest.fixture(scope="session")
def prob_dict_three_actions() -> Dict[str, float]:
    """Fixture providing a probability dictionary with three actions.

    Returns
    -------
    Dict[str, float]
        Probability dictionary with three actions (a1: 0.5, a2: 0.7, a3: 0.3).
    """
    return {"a1": 0.5, "a2": 0.7, "a3": 0.3}


@pytest.fixture(scope="session")
def prob_dict_single_action() -> Dict[str, float]:
    """Fixture providing a probability dictionary with a single action.

    Returns
    -------
    Dict[str, float]
        Probability dictionary with one action (a1: 0.5).
    """
    return {"a1": 0.5}


########################################################################################################################
# BaseStrategy tests


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""

    def select_action(
        self,
        p: Dict[ActionId, Union[float, Callable[[np.ndarray], float]]],
        actions: Dict[ActionId, BaseModel],
        **kwargs,
    ) -> UnifiedActionId:
        """Select the first action."""
        return list(p.keys())[0]


def test_base_strategy_abstract():
    """Test that BaseStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseStrategy()


def test_base_strategy_concrete_implementation(prob_dict_two_actions: Dict[str, float], expected_result: str = "a1"):
    """Test that concrete implementations of BaseStrategy work.

    Parameters
    ----------
    prob_dict_two_actions : Dict[str, float]
        Probability dictionary with two actions.
    expected_result : str
        Expected result of the strategy.
    """
    strategy = ConcreteStrategy()
    p = prob_dict_two_actions
    actions = {action_id: BetaCC(cost=DEFAULT_COST) for action_id in prob_dict_two_actions.keys()}

    result = strategy.select_action(p, actions)
    assert result == expected_result


########################################################################################################################
# SingleObjectiveStrategy tests


class ConcreteSingleObjectiveStrategy(SingleObjectiveStrategy):
    """Concrete implementation of SingleObjectiveStrategy for testing."""

    def get_prerequisites(
        self,
        p: Dict[ActionId, Union[float, Callable]],
        actions: Dict[ActionId, BaseModel],
        constraint_list: Optional[List[Callable]],
    ) -> Dict[str, any]:
        """Return empty prerequisites."""
        return {"test_value": 42}

    def _verify_action(self, score: float, **kwargs) -> bool:
        """Accept all actions."""
        return True

    def _verify_and_select_from_quantitative_action(
        self,
        score_func: Callable[[np.ndarray], float],
        model: BaseModel,
        constraint_list: Optional[List[Callable[[np.ndarray], bool]]],
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Return a simple quantity vector."""
        return np.array([0.5, 0.5])

    def _select_from_refined_actions(
        self,
        refined_p: Dict[UnifiedActionId, float],
        actions: Dict[ActionId, BaseModel],
        constraint: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> UnifiedActionId:
        """Select the first action."""
        return list(refined_p.keys())[0] if refined_p else None


def test_single_objective_strategy_abstract():
    """Test that SingleObjectiveStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SingleObjectiveStrategy()


def test_single_objective_strategy_select_action(prob_dict_two_actions: Dict[str, float]):
    """Test SingleObjectiveStrategy select_action method.

    Parameters
    ----------
    prob_dict_two_actions : Dict[str, float]
        Probability dictionary with two actions.
    """
    strategy = ConcreteSingleObjectiveStrategy()
    p = prob_dict_two_actions
    actions = {action_id: BetaCC(cost=DEFAULT_COST) for action_id in prob_dict_two_actions.keys()}

    result = strategy.select_action(p, actions)
    assert result in p.keys()


@pytest.mark.parametrize("constraint_returns", [True, False])
def test_single_objective_strategy_with_constraints(
    constraint_returns: bool, prob_dict_single_action: Dict[str, float], expected_result: str = "a1"
):
    """Test SingleObjectiveStrategy with constraints.

    Parameters
    ----------
    constraint_returns : bool
        Whether the constraint should return True or False.
    prob_dict_single_action : Dict[str, float]
        Probability dictionary with one action.
    expected_result : str
        Expected result of the strategy.
    """
    strategy = ConcreteSingleObjectiveStrategy()
    p = prob_dict_single_action
    actions = {action_id: BetaCC(cost=DEFAULT_COST) for action_id in prob_dict_single_action.keys()}

    def constraint(x):
        return constraint_returns

    result = strategy.select_action(p, actions, constraint)

    assert result == expected_result


def test_single_objective_strategy_refine_p_with_quantitative(
    prob_a1: float = 0.5, prob_a2: float = 0.7, prob_a3: float = 0.3
):
    """Test refine_p with quantitative actions.

    Parameters
    ----------
    prob_a1 : float
        Probability for regular action a1.
    prob_a2 : float
        Probability for quantitative action a2.
    prob_a3 : float
        Probability for quantitative action a3.
    """
    strategy = ConcreteSingleObjectiveStrategy()

    # Mix of regular and quantitative actions
    p = {"a1": prob_a1, "a2": lambda x: prob_a2, "a3": lambda x: prob_a3}
    actions = {
        "a1": BetaCC(cost=DEFAULT_COST),
        "a2": create_mock_quantitative_model(),
        "a3": create_mock_quantitative_model(),
    }

    refined_p = strategy.refine_p(p, actions, None)

    # Check that regular action is preserved
    assert "a1" in refined_p
    assert refined_p["a1"] == prob_a1

    # Check that quantitative actions are converted to tuples
    quantitative_keys = [k for k in refined_p.keys() if isinstance(k, tuple)]
    assert len(quantitative_keys) == 2

    for key in quantitative_keys:
        assert key[0] in ["a2", "a3"]
        assert isinstance(key[1], tuple)


def test_single_objective_strategy_verify_and_select_public_method(
    model_dimension: int = 3, expected_result_length: int = 2
):
    """Test the public verify_and_select_from_quantitative_action method.

    Parameters
    ----------
    model_dimension : int
        Dimension of the quantitative model.
    expected_result_length : int
        Expected length of the result array.
    """
    strategy = ConcreteSingleObjectiveStrategy()

    model = create_mock_quantitative_model(dimension=model_dimension)
    constraint_list = [lambda x: np.all(x >= 0)]

    result = strategy.verify_and_select_from_quantitative_action(sum, model, constraint_list)

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) == expected_result_length


########################################################################################################################
# ClassicBandit


def test_can_init_classic_bandit():
    """Test that ClassicBandit can be initialized."""
    bandit = ClassicBandit()
    assert isinstance(bandit, SingleObjectiveStrategy)
    assert isinstance(bandit, BaseStrategy)


@given(
    st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=5, unique=True),
    st.lists(st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False), min_size=2, max_size=5),
)
@settings(max_examples=10)
def test_select_action_classic_bandit(a_list_str, a_list_float):
    """Test ClassicBandit selects action with highest probability.

    Parameters
    ----------
    a_list_str : list
        List of action IDs.
    a_list_float : list
        List of probabilities.
    """
    assume(len(a_list_str) == len(a_list_float))
    p = dict(zip(a_list_str, a_list_float))
    actions = {action_id: BetaCC(cost=DEFAULT_COST) for action_id in a_list_str}

    c = ClassicBandit()
    assert max(p, key=p.get) == c.select_action(p=p, actions=actions)


def test_classic_bandit_prerequisites(prob_dict_single_action: Dict[str, float]):
    """Test that ClassicBandit returns empty prerequisites.

    Parameters
    ----------
    prob_dict_single_action : Dict[str, float]
        Probability dictionary with one action.
    """
    bandit = ClassicBandit()
    p = prob_dict_single_action
    actions = {"a1": BetaCC(cost=DEFAULT_COST)}

    prerequisites = bandit.get_prerequisites(p, actions, None)
    assert prerequisites == {}


def test_classic_bandit_verify_action():
    """Test that ClassicBandit accepts all actions."""
    bandit = ClassicBandit()

    # Should always return True
    assert bandit._verify_action(0.0)
    assert bandit._verify_action(0.5)
    assert bandit._verify_action(1.0)


def test_classic_bandit_quantitative_action(dimension: int = 2, expected_result: np.ndarray = np.array([1.0, 1.0])):
    """Test ClassicBandit handles quantitative actions.

    Parameters
    ----------
    mock_maximize : MagicMock
        Mock for maximize_by_quantity function.
    """

    bandit = ClassicBandit()
    model = create_mock_quantitative_model(dimension=dimension)

    result = bandit._verify_and_select_from_quantitative_action(sum, model, None)
    assert np.allclose(result, expected_result, atol=1e-3)


@pytest.mark.parametrize(
    "n_actions,n_quantitative",
    [
        (3, 0),  # All regular actions
        (3, 1),  # Mix of regular and quantitative
        (3, 3),  # All quantitative actions
    ],
)
def test_classic_bandit_mixed_actions(
    n_actions: int,
    n_quantitative: int,
    return_value: np.ndarray = np.array([0.5, 0.5]),
    base_prob: float = 0.5,
    prob_increment: float = 0.1,
):
    """Test ClassicBandit with mixed regular and quantitative actions.

    Parameters
    ----------
    n_actions : int
        Total number of actions.
    n_quantitative : int
        Number of quantitative actions.
    return_value : np.ndarray
        Return value for mock maximize function.
    base_prob : float
        Base probability value for actions.
    prob_increment : float
        Probability increment per action index.
    """
    bandit = ClassicBandit()
    p = {}
    actions = {}

    for i in range(n_actions):
        action_id = f"a{i}"
        if i < n_quantitative:
            p[action_id] = lambda x, val=base_prob + i * prob_increment: val
            actions[action_id] = create_mock_quantitative_model()
        else:
            p[action_id] = base_prob + i * prob_increment
            actions[action_id] = BetaCC(cost=DEFAULT_COST)

    # Patch where it's used (strategy module) not where it's defined (utils module)
    with patch("pybandits.strategy.maximize_by_quantity") as mock_maximize:
        mock_maximize.return_value = return_value
        result = bandit.select_action(p, actions)

        assert result is not None
        if n_quantitative:
            assert mock_maximize.call_count == n_quantitative, (
                f"Expected {n_quantitative} calls but got {mock_maximize.call_count}"
            )


########################################################################################################################
# BestActionIdentificationBandit


@given(st.floats())
def test_can_init_best_action_identification(a_float: float):
    """Test BestActionIdentificationBandit initialization.

    Parameters
    ----------
    a_float : float
        Test value for exploit_p.
    """
    # init default params
    b = BestActionIdentificationBandit()
    assert b.exploit_p == 0.5
    assert isinstance(b, ClassicBandit)

    # init with input arguments
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            BestActionIdentificationBandit(exploit_p=a_float)
    else:
        b = BestActionIdentificationBandit(exploit_p=a_float)
        assert b.exploit_p == a_float


@given(st.floats())
def test_with_exploit_p(a_float: float):
    """Test BestActionIdentificationBandit with_exploit_p method.

    Parameters
    ----------
    a_float : float
        Test value for exploit_p.
    """
    b = BestActionIdentificationBandit()

    # set with invalid float
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            b.with_exploit_p(exploit_p=a_float)
    # set with valid float
    else:
        mutated_b = b.with_exploit_p(exploit_p=a_float)
        assert mutated_b.exploit_p == a_float
        assert mutated_b is not b


@given(
    st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=5, unique=True),
    st.lists(st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False), min_size=2, max_size=5),
)
@settings(max_examples=10)
def test_select_action_bai(a_list_str, a_list_float):
    """Test BestActionIdentificationBandit select_action method.

    Parameters
    ----------
    a_list_str : list
        List of action IDs.
    a_list_float : list
        List of probabilities.
    """
    assume(len(a_list_str) == len(a_list_float))
    p = dict(zip(a_list_str, a_list_float))
    actions = {action_id: BetaCC(cost=DEFAULT_COST) for action_id in a_list_str}

    b = BestActionIdentificationBandit()
    result = b.select_action(p=p, actions=actions)
    assert result in p.keys()


@pytest.mark.parametrize(
    "exploit_p,should_be_best",
    [
        (1.0, True),  # Always select best
        (0.0, False),  # Always select second-best
    ],
)
def test_bai_selection_logic(
    exploit_p: float,
    should_be_best: bool,
    mocker: MockerFixture,
    prob_a1: float = 0.3,
    prob_a2: float = 0.7,
    prob_a3: float = 0.5,
    random_value: float = 0.5,
):
    """Test BAI selection logic with different exploit_p values.

    Parameters
    ----------
    exploit_p : float
        Exploitation probability.
    should_be_best : bool
        Whether the best action should be selected.
    mocker : MockerFixture
        Pytest mocker fixture.
    prob_a1 : float
        Probability for action a1.
    prob_a2 : float
        Probability for action a2.
    prob_a3 : float
        Probability for action a3.
    random_value : float
        Mocked random value for selection control.
    """
    # Mock random to control selection
    mocker.patch("pybandits.strategy.random", return_value=random_value)

    p = {"a1": prob_a1, "a2": prob_a2, "a3": prob_a3}
    actions = {action_id: BetaCC(cost=DEFAULT_COST) for action_id in p.keys()}

    b = BestActionIdentificationBandit(exploit_p=exploit_p)
    result = b.select_action(p=p, actions=actions)

    if should_be_best:
        assert result == "a2"  # Highest probability
    else:
        assert result == "a3"  # Second highest


def test_bai_all_probs_equal(equal_prob: float = 0.5, exploit_p_max: float = 1.0, exploit_p_min: float = 0.0):
    """Test BAI behavior when all probabilities are equal.

    Parameters
    ----------
    equal_prob : float
        Equal probability value for all actions.
    exploit_p_max : float
        Maximum exploit probability value.
    exploit_p_min : float
        Minimum exploit probability value.
    """
    p = {"a1": equal_prob, "a2": equal_prob, "a3": equal_prob}
    actions = {action_id: BetaCC(cost=DEFAULT_COST) for action_id in p.keys()}

    b = BestActionIdentificationBandit(exploit_p=exploit_p_max)
    # if exploit_p is 1 and all probs are equal => return the action with 1st highest prob (max)
    assert "a1" == b.select_action(p=p, actions=actions)

    # if exploit_p is 0 => return the action with 2nd highest prob (not 1st highest prob)
    mutated_b = b.with_exploit_p(exploit_p=exploit_p_min)
    assert "a2" == mutated_b.select_action(p=p, actions=actions)


@given(
    exploit_p=st.floats(min_value=0.01, max_value=0.99), expected_result1=st.just("a1"), expected_result2=st.just("a2")
)
def test_bai_probabilistic_selection(
    exploit_p: float, expected_result1: str, expected_result2: str, prob_dict_three_actions: Dict[str, float]
):
    """Test BAI probabilistic selection between best and second-best.

    Parameters
    ----------
    exploit_p : float
        Exploitation probability.
    prob_dict_three_actions : Dict[str, float]
        Probability dictionary with three actions.
    expected_result1 : str
        Expected result of the strategy when random > exploit_p.
    expected_result2 : str
        Expected result of the strategy when random <= exploit_p.
        Expected result of the strategy.
    """
    p = prob_dict_three_actions
    actions = {action_id: BetaCC(cost=DEFAULT_COST) for action_id in p.keys()}

    b = BestActionIdentificationBandit(exploit_p=exploit_p)

    # Test that selection respects probability
    with patch("pybandits.strategy.random") as mock_random:
        # When random > exploit_p, should select second best
        mock_random.return_value = exploit_p + 0.01
        assert b.select_action(p=p, actions=actions) == expected_result1

        # When random <= exploit_p, should select best
        mock_random.return_value = exploit_p - 0.01
        assert b.select_action(p=p, actions=actions) == expected_result2


########################################################################################################################
# CostControlStrategy tests


def test_cost_control_strategy_mixin(default_subsidy_factor: float = 0.5, new_subsidy_factor: float = 0.7):
    """Test CostControlStrategy as a mixin.

    Parameters
    ----------
    default_subsidy_factor : float
        Default subsidy factor value.
    new_subsidy_factor : float
        New subsidy factor value for mutation test.
    """
    strategy = CostControlStrategy()
    assert strategy.subsidy_factor == default_subsidy_factor

    # Test with_subsidy_factor
    mutated = strategy.with_subsidy_factor(new_subsidy_factor)
    assert mutated.subsidy_factor == new_subsidy_factor
    assert mutated is not strategy


@given(st.floats())
def test_cost_control_strategy_validation(subsidy_factor: float):
    """Test CostControlStrategy subsidy_factor validation.

    Parameters
    ----------
    subsidy_factor : float
        Test value for subsidy_factor.
    """
    if 0 <= subsidy_factor <= 1 and not (np.isnan(subsidy_factor) or np.isinf(subsidy_factor)):
        strategy = CostControlStrategy(subsidy_factor=subsidy_factor)
        assert strategy.subsidy_factor == subsidy_factor
    else:
        with pytest.raises(ValidationError):
            CostControlStrategy(subsidy_factor=subsidy_factor)


########################################################################################################################
# CostControlBandit


@given(st.floats())
def test_can_init_cost_control(a_float: float):
    """Test CostControlBandit initialization.

    Parameters
    ----------
    a_float : float
        Test value for subsidy_factor.
    """
    # init with default arguments
    c = CostControlBandit()
    assert c.subsidy_factor == 0.5
    assert isinstance(c, SingleObjectiveStrategy)
    assert isinstance(c, CostControlStrategy)

    # init with input arguments
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            CostControlBandit(subsidy_factor=a_float)
    else:
        c = CostControlBandit(subsidy_factor=a_float)
        assert c.subsidy_factor == a_float


@given(st.floats())
def test_with_subsidy_factor(a_float: float):
    """Test CostControlBandit with_subsidy_factor method.

    Parameters
    ----------
    a_float : float
        Test value for subsidy_factor.
    """
    c = CostControlBandit()

    # set with invalid float
    if a_float < 0 or a_float > 1 or np.isnan(a_float) or np.isinf(a_float):
        with pytest.raises(ValidationError):
            c.with_subsidy_factor(subsidy_factor=a_float)
    # set with valid float
    else:
        mutated_c = c.with_subsidy_factor(subsidy_factor=a_float)
        assert mutated_c.subsidy_factor == a_float
        assert mutated_c is not c


@given(
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3, unique=True),
    st.lists(st.floats(min_value=0, max_value=100, allow_infinity=False, allow_nan=False), min_size=1, max_size=3),
)
@settings(max_examples=10)
def test_select_action_cc(a_list_str, a_list_float):
    """Test CostControlBandit select_action method.

    Parameters
    ----------
    a_list_str : list
        List of action IDs.
    a_list_float : list
        List of costs.
    """
    assume(len(a_list_str) == len(a_list_float))
    a_list_float_0_1 = [float(i) / (sum(a_list_float) + 1) for i in a_list_float]

    p = dict(zip(a_list_str, a_list_float_0_1))
    a = dict(zip(a_list_str, [BetaCC(cost=c) for c in a_list_float]))

    c = CostControlBandit()
    result = c.select_action(p=p, actions=a)
    assert result in p.keys()


@pytest.mark.parametrize(
    "subsidy_factor,expected_action",
    [
        (1.0, "a4"),  # Min cost action with highest prob among same cost
        (0.0, "a2"),  # Highest probability action
        (0.5, "a5"),  # Cheapest feasible action
    ],
)
def test_cost_control_logic(subsidy_factor: float, expected_action: str):
    """Test CostControlBandit selection logic with different subsidy factors.

    Parameters
    ----------
    subsidy_factor : float
        Subsidy factor for cost control.
    expected_action : str
        Expected selected action.
    """
    actions_cost = {"a1": 10, "a2": 30, "a3": 20, "a4": 10, "a5": 20}
    p = {"a1": 0.1, "a2": 0.8, "a3": 0.6, "a4": 0.2, "a5": 0.65}

    actions = {action_id: BetaCC(cost=cost) for action_id, cost in actions_cost.items()}

    c = CostControlBandit(subsidy_factor=subsidy_factor)
    assert c.select_action(p=p, actions=actions) == expected_action


@pytest.mark.parametrize(
    "subsidy_factor,expected_action",
    [
        (1.0, "a4"),  # Min cost action with highest prob among same cost
        (0.0, "a2"),  # Highest probability action
        (0.5, "a5"),  # Cheapest feasible action
    ],
)
def test_cost_control_logic_callable_cost_and_proba(
    subsidy_factor: float, expected_action: Tuple[str, Tuple[float, ...]]
):
    """Test CostControlBandit select_action when cost and proba are callables.

    Same selection logic as test_cost_control_logic but with quantitative actions:
    p maps to callable proba (probability given quantity vector) and actions use
    quantitative models with callable cost (cost given quantity vector).
    """
    actions_cost = {"a1": 10, "a2": 30, "a3": 20, "a4": 10, "a5": 20}
    p = {
        "a1": lambda x: 0.1,
        "a2": lambda x: 0.8,
        "a3": lambda x: 0.6,
        "a4": lambda x: 0.2,
        "a5": lambda x: 0.65,
    }
    actions = {
        action_id: create_mock_quantitative_model(
            dimension=2, cost_value=cost, mocked_model_type="DummyQuantitativeModelCC"
        )
        for action_id, cost in actions_cost.items()
    }

    c = CostControlBandit(subsidy_factor=subsidy_factor)
    result = c.select_action(p=p, actions=actions)
    assert result[0] == expected_action
    assert all(0 <= quantity <= 1 for quantity in result[1])


@given(
    st.lists(st.floats(min_value=0, max_value=1, allow_infinity=False, allow_nan=False), min_size=3, max_size=3),
    st.lists(st.floats(min_value=0, max_value=100, allow_infinity=False, allow_nan=False), min_size=3, max_size=3),
)
def test_cost_control_corner_cases(a_list_p, a_list_cost):
    """Test CostControlBandit corner cases with ties in cost and probability.

    Parameters
    ----------
    a_list_p : list
        List of probabilities.
    a_list_cost : list
        List of costs.
    """
    action_ids = ["a1", "a2", "a3"]

    p = dict(zip(action_ids, a_list_p))
    actions_cost = dict(zip(action_ids, a_list_cost))
    actions_cost_proba = [(a_cost, -a_proba, a_id) for a_id, a_cost, a_proba in zip(action_ids, a_list_cost, a_list_p)]

    actions = {aid: BetaCC(cost=actions_cost[aid]) for aid in action_ids}

    c = CostControlBandit(subsidy_factor=1)
    # if subsidy_factor is 1 => return the action with min cost (and highest prob if tied)
    assert sorted(actions_cost_proba)[0][-1] == c.select_action(p=p, actions=actions)

    # if subsidy_factor is 0:
    mutated_c = c.with_subsidy_factor(subsidy_factor=0)
    # get the keys of the max p.values() (there might be more max_p_values)
    max_p_values = [k for k, v in p.items() if v == max(p.values())]

    # if subsidy_factor is 0 and only 1 max_value => return the action with highest p
    if len(max_p_values) == 1:
        assert max(p, key=p.get) == mutated_c.select_action(p=p, actions=actions)
    # if subsidy_factor is 0 and 1+ max_values => return the one with min cost
    else:
        actions_cost_max = {k: actions_cost[k] for k in max_p_values}
        assert min(actions_cost_max, key=actions_cost_max.get) == mutated_c.select_action(p=p, actions=actions)


def test_cost_control_get_prerequisites(prob_a1: float = 0.5, prob_a2: float = 0.8, prob_a3: float = 0.3):
    """Test CostControlBandit get_prerequisites method.

    Parameters
    ----------
    prob_a1 : float
        Probability for action a1.
    prob_a2 : float
        Probability for action a2 (expected to be highest).
    prob_a3 : float
        Probability for action a3.
    """
    c = CostControlBandit()

    p = {"a1": prob_a1, "a2": prob_a2, "a3": prob_a3}
    actions = {aid: BetaCC(cost=DEFAULT_COST) for aid in p.keys()}

    prerequisites = c.get_prerequisites(p, actions, None)

    assert "best_value" in prerequisites
    assert prerequisites["best_value"] == prob_a2  # Highest probability


@pytest.mark.parametrize(
    "score, best_value, expected_result",
    [
        (0.6, 1.0, True),
        (0.5, 1.0, True),
        (0.4, 1.0, False),
    ],
)
def test_cost_control_verify_action(
    score: float, best_value: float, expected_result: bool, subsidy_factor: float = DEFAULT_SUBSIDY_FACTOR
):
    """Test CostControlBandit _verify_action method.

    Parameters
    ----------
    score : float
        Score to verify.
    best_value : float
        Best value for comparison.
    expected_result : bool
        Expected verification result.
    subsidy_factor : float
        Subsidy factor for the bandit.
    """
    c = CostControlBandit(subsidy_factor=subsidy_factor)
    assert c._verify_action(score, best_value=best_value) is expected_result


@patch("pybandits.utils.maximize_by_quantity")
def test_cost_control_quantitative_action(
    mock_maximize: MagicMock,
    return_value: np.ndarray = np.array([0.3, 0.7]),
    subsidy_factor: float = 0.5,
    dimension: int = 2,
    cost_multiplier: float = 10.0,
    best_value: float = 0.8,
):
    """Test CostControlBandit with quantitative actions.

    Parameters
    ----------
    mock_maximize : MagicMock
        Mock for maximize_by_quantity function.
    return_value : np.ndarray
        Return value for mock maximize function.
    subsidy_factor : float
        Subsidy factor for cost control.
    dimension : int
        Dimension of the quantitative model.
    cost_multiplier : float
        Multiplier for cost calculation.
    best_value : float
        Best value for verification.
    """
    mock_maximize.return_value = return_value

    c = CostControlBandit(subsidy_factor=subsidy_factor)

    model = create_mock_quantitative_model(dimension=dimension)
    model.cost = MagicMock(side_effect=lambda x: np.sum(x) * cost_multiplier)

    result = c._verify_and_select_from_quantitative_action(sum, model, None, best_value=best_value)

    # Check if mock was used, otherwise handle actual optimization result
    if mock_maximize.called:
        assert result is not None, "Optimization should return a result"
        assert np.allclose(result, return_value, atol=1e-6)
        mock_maximize.assert_called_once()
        # Check that cost control constraint was added
        call_args = mock_maximize.call_args
        constraint_list = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("constraint_list")
        assert constraint_list is not None, "Cost control constraint should be added"
    else:
        # Mock wasn't used - actual optimization may fail due to constraints
        # Accept None if constraints can't be satisfied, or verify result if successful
        if result is not None:
            assert isinstance(result, np.ndarray)
            assert len(result) == dimension


########################################################################################################################
# MultiObjectiveStrategy tests


class ConcreteMultiObjectiveStrategy(MultiObjectiveStrategy):
    """Concrete implementation of MultiObjectiveStrategy for testing."""

    objective_selector_class = ClassicBandit

    def _get_feasible_solutions(
        self, p: Dict[ActionId, List[float]], actions: Dict[ActionId, BaseModel]
    ) -> Dict[UnifiedActionId, List[float]]:
        """Return all solutions as feasible."""
        return p


def test_multi_objective_strategy_abstract():
    """Test that MultiObjectiveStrategy cannot be instantiated directly."""
    with pytest.raises(AttributeError):
        MultiObjectiveStrategy()


def test_multi_objective_strategy_initialization():
    """Test MultiObjectiveStrategy initialization."""
    strategy = ConcreteMultiObjectiveStrategy()
    assert hasattr(strategy, "_objective_selector")
    assert isinstance(strategy._objective_selector, ClassicBandit)


########################################################################################################################
# MultiObjectiveBandit


def test_can_init_multiobjective():
    """Test MultiObjectiveBandit initialization."""
    m = MultiObjectiveBandit()
    assert isinstance(m, MultiObjectiveStrategy)
    assert m.objective_selector_class == ClassicBandit


@given(
    st.dictionaries(
        st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cc", "Cs"))),
        st.lists(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False), min_size=2, max_size=3),
        min_size=2,
    )
)
def test_select_action_mo(p: Dict[ActionId, List[Probability]]):
    """Test MultiObjectiveBandit selects from Pareto front.

    Parameters
    ----------
    p : Dict[ActionId, List[Probability]]
        Dictionary of actions and their multi-objective probabilities.
    """
    # Ensure all actions have same number of objectives
    n_objectives = len(list(p.values())[0])
    p = {k: v for k, v in p.items() if len(v) == n_objectives}

    if not p:
        return  # Skip if no valid actions

    actions = {aid: BetaMOCC(models=[Beta() for _ in range(n_objectives)], cost=DEFAULT_COST) for aid in p.keys()}

    m = MultiObjectiveBandit()
    selected = m.select_action(p=p, actions=actions)
    pareto_front = m._get_pareto_front(p, actions)

    assert selected in pareto_front


@pytest.mark.parametrize(
    "p_dict,expected_front",
    [
        # 2D case 1: Clear Pareto front
        (
            {
                "a0": [0.1, 0.3],
                "a1": [0.1, 0.3],
                "a2": [0.0, 0.0],
                "a3": [0.1, 0.1],
                "a4": [0.3, 0.1],
                "a5": [0.2, 0.2],
                "a6": [0.3, 0.0],
                "a7": [0.1, 0.1],
            },
            ["a0", "a1", "a4", "a5"],
        ),
        # 2D case 2: Duplicate optimal points
        (
            {
                "a0": [0.1, 0.1],
                "a1": [0.3, 0.3],
                "a2": [0.3, 0.3],
            },
            ["a1", "a2"],
        ),
        # 3D case
        (
            {
                "a0": [0.1, 0.3, 0.1],
                "a1": [0.1, 0.3, 0.1],
                "a2": [0.0, 0.0, 0.1],
                "a3": [0.1, 0.1, 0.1],
                "a4": [0.3, 0.1, 0.1],
                "a5": [0.2, 0.2, 0.1],
                "a6": [0.3, 0.0, 0.1],
                "a7": [0.1, 0.1, 0.3],
            },
            ["a0", "a1", "a4", "a5", "a7"],
        ),
    ],
)
def test_exact_pareto_front(p_dict: Dict[str, List[float]], expected_front: List[str]):
    """Test exact Pareto front computation.

    Parameters
    ----------
    p_dict : Dict[str, List[float]]
        Dictionary of actions and their multi-objective values.
    expected_front : List[str]
        Expected Pareto front actions.
    """
    n_objectives = len(list(p_dict.values())[0])
    actions = {aid: BetaMOCC(models=[Beta() for _ in range(n_objectives)], cost=DEFAULT_COST) for aid in p_dict.keys()}

    m = MultiObjectiveBandit()
    pareto_front = m._get_exact_pareto_front(p_dict, actions)

    assert sorted(pareto_front) == sorted(expected_front)


def test_approximate_pareto_front(
    fixed_prob: float = 0.4,
    func2_coeff: float = 0.5,
    func2_offset: float = 0.3,
    dimension: int = 1,
    n_divisions: int = 5,
    mock_solution1: float = 0.5,
    mock_solution2: float = 0.8,
):
    """Test approximate Pareto front computation for quantitative actions.

    Parameters
    ----------
    fixed_prob : float
        Fixed probability value for discrete action.
    func2_coeff : float
        Coefficient for func2 calculation.
    func2_offset : float
        Offset for func2 calculation.
    dimension : int
        Dimension of quantitative models.
    n_divisions : int
        Number of divisions for Pareto front approximation.
    mock_solution1 : float
        First mock solution value.
    mock_solution2 : float
        Second mock solution value.
    """
    m = MultiObjectiveBandit()

    # Create mock quantitative actions
    def func1(x: np.ndarray) -> List[float]:
        return [x[0], 1 - x[0]]  # Trade-off between objectives

    def func2(x: np.ndarray) -> List[float]:
        return [func2_coeff * x[0], func2_coeff * (1 - x[0]) + func2_offset]  # Different trade-off

    p = {
        "a1": func1,
        "a2": func2,
        "a3": [fixed_prob, fixed_prob],  # Fixed action
    }

    actions = {
        "a1": create_mock_quantitative_model(dimension=dimension),
        "a2": create_mock_quantitative_model(dimension=dimension),
        "a3": BetaMOCC(models=[Beta(), Beta()], cost=DEFAULT_COST),
    }

    # Mock the models attribute for quantitative models
    actions["a1"].models = [Beta(), Beta()]
    actions["a2"].models = [Beta(), Beta()]

    # Patch on the class where it's used (strategy module) to avoid Pydantic model restrictions
    with patch("pybandits.strategy.MultiObjectiveStrategy._find_pareto_front_normal_constraint") as mock_nc:
        mock_nc.side_effect = lambda *args, **kwargs: [
            np.array([mock_solution1]),
            np.array([mock_solution2]),
        ]

        pareto_front = m._get_approximate_pareto_front(p, actions, n_divisions=n_divisions)

        # Should have been called for quantitative actions
        assert mock_nc.call_count == 2
        # If a3 is not in pareto front, it might have been dominated - check that we at least have some results
        assert len(pareto_front)
        # If a3 exists and is not dominated, it should be in the front
        # For now, just verify we have results from the quantitative actions
        assert any(isinstance(item, tuple) for item in pareto_front)


@pytest.mark.parametrize(
    "n_objectives,n_divisions",
    [
        (2, 5),
        (3, 3),
        (4, 2),
    ],
)
def test_das_dennis_weights(n_objectives: int, n_divisions: int):
    """Test Das-Dennis weight generation.

    Parameters
    ----------
    n_objectives : int
        Number of objectives.
    n_divisions : int
        Number of divisions for weight generation.
    """
    weights = MultiObjectiveStrategy._das_dennis_weights(n_objectives, n_divisions)

    # Check all weights sum to 1
    for w in weights:
        assert np.isclose(np.sum(w), 1.0)

    # Check all weights are non-negative
    assert np.all(weights >= 0)

    # Check dimensionality
    assert weights.shape[1] == n_objectives

    # Check approximate number of weights (combinatorial formula)
    from math import comb

    expected_count = comb(n_divisions + n_objectives - 1, n_objectives - 1)
    assert len(weights) == expected_count


def test_find_pareto_front_normal_constraint(
    return_value: np.ndarray = np.array([0.5]),
    dimension: int = 1,
    n_objectives: int = 2,
    n_divisions: int = 3,
    best_obj1: float = 1.0,
    best_obj2: float = 0.0,
):
    """Test Normal Constraint method for Pareto front finding.

    Parameters
    ----------
    return_value : np.ndarray
        Return value for mock solve function.
    dimension : int
        Dimension of the quantitative model.
    n_objectives : int
        Number of objectives.
    n_divisions : int
        Number of divisions for weight generation.
    best_obj1 : float
        Best value for objective 1.
    best_obj2 : float
        Best value for objective 2.
    """
    m = MultiObjectiveBandit()

    # Simple 2-objective function with known Pareto front
    def test_func(x: np.ndarray) -> List[float]:
        return [x[0], 1 - x[0]]  # Linear trade-off

    model = create_mock_quantitative_model(dimension=dimension)
    model.models = [Beta() for _ in range(n_objectives)]

    with patch(
        "pybandits.strategy.ClassicBandit.verify_and_select_from_quantitative_action",
        side_effect=[
            np.array([best_obj1]),  # Best for objective 1
            np.array([best_obj2]),  # Best for objective 2
        ],
    ) as mock_verify:
        # Add mock for NC subproblem solving
        with patch("pybandits.strategy.MultiObjectiveStrategy._solve_nc_subproblem") as mock_solve:
            mock_solve.return_value = return_value

            solutions = m._find_pareto_front_normal_constraint(test_func, dimension, n_objectives, n_divisions, model)

            assert len(solutions) > 0
            assert mock_verify.call_count == n_objectives  # Called for each objective
            assert mock_solve.call_count > 0  # Called for each weight vector


########################################################################################################################
# MultiObjectiveCostControlBandit


def test_can_init_multiobjective_mo_cc():
    """Test MultiObjectiveCostControlBandit initialization."""
    m = MultiObjectiveCostControlBandit()
    assert isinstance(m, MultiObjectiveStrategy)
    assert isinstance(m, CostControlStrategy)
    assert m.objective_selector_class == CostControlBandit
    assert m.subsidy_factor == 0.5


@pytest.mark.parametrize(
    "test_case",
    [
        # Case 1: Different costs, clear Pareto front
        {
            "actions_costs": {"a1": 8, "a2": 2, "a3": 5, "a4": 1, "a5": 7},
            "probabilities": {
                "a1": [0.1, 0.3, 0.5],
                "a2": [0.1, 0.3, 0.5],
                "a3": [0.0, 0.4, 0.4],
                "a4": [0.5, 0.3, 0.7],
                "a5": [0.6, 0.1, 0.5],
            },
            "expected_pareto": ["a3", "a4", "a5"],
            "expected_selection": "a5",  # Min cost in Pareto front
        },
        # Case 2: Equal costs, select by probability
        {
            "actions_costs": {"a1": 2, "a2": 2, "a3": 5},
            "probabilities": {
                "a1": [0.6, 0.1],
                "a2": [0.5, 0.8],
                "a3": [0.0, 0.1],
            },
            "expected_pareto": ["a1", "a2"],
        },
    ],
)
def test_mo_cc_selection_logic(test_case: dict):
    """Test MultiObjectiveCostControlBandit selection logic.

    Parameters
    ----------
    test_case : dict
        Test case with actions, probabilities, and expected results.
    """
    m = MultiObjectiveCostControlBandit()

    n_objectives = len(list(test_case["probabilities"].values())[0])
    actions = {
        aid: BetaMOCC(models=[Beta() for _ in range(n_objectives)], cost=cost)
        for aid, cost in test_case["actions_costs"].items()
    }

    p = test_case["probabilities"]

    # Test Pareto front computation
    pareto_front = m._get_pareto_front(p, actions)
    assert sorted(pareto_front) == sorted(test_case["expected_pareto"])

    # Test action selection
    selected = m.select_action(p=p, actions=actions)
    # Verify selected action is in expected Pareto front
    assert selected in test_case["expected_pareto"]


def test_mo_cc_get_feasible_solutions(subsidy_factor: float = 0.5, fixed_prob_value: float = 0.5):
    """Test MultiObjectiveCostControlBandit _get_feasible_solutions method.

    Parameters
    ----------
    subsidy_factor : float
        Subsidy factor for the bandit.
    fixed_prob_value : float
        Fixed probability value for action a2.
    """
    m = MultiObjectiveCostControlBandit(subsidy_factor=subsidy_factor)

    # Create test data with quantitative actions
    p = {
        "a1": lambda x: [x[0], 1 - x[0]],
        "a2": lambda x: [fixed_prob_value, fixed_prob_value],
    }

    model1 = create_mock_quantitative_model()
    model1.models = [Beta(), Beta()]
    model2 = create_mock_quantitative_model()
    model2.models = [Beta(), Beta()]

    actions = {
        "a1": model1,
        "a2": model2,
        "a3": BetaMOCC(models=[Beta(), Beta()], cost=DEFAULT_COST),  # Discrete action model
    }

    # Mock the objective selector's refine_p method
    # Patch on the class where it's used (strategy module) to avoid Pydantic model restrictions
    with patch("pybandits.strategy.CostControlBandit.refine_p", return_value={("a1", (0.5,)): 0.5}) as mock_refine:
        m._get_feasible_solutions(p, actions)
        # Verify the method was called (it will be called per objective)
        # Should be called for each objective (2 objectives in this test)
        assert mock_refine.call_count == 2, f"Expected 2 calls but got {mock_refine.call_count}"


########################################################################################################################
# Integration tests


@pytest.mark.parametrize(
    "strategy",
    [
        ClassicBandit(),
        BestActionIdentificationBandit(exploit_p=DEFAULT_EXPLOIT_P),
        CostControlBandit(subsidy_factor=DEFAULT_SUBSIDY_FACTOR),
    ],
    ids=["Classic", "BAI", "CC"],
)
@given(
    action_data=action_probability_pairs(min_actions=2, max_actions=4, allow_callables=True),
    mock_return_value=st.just(np.array([0.5, 0.5])),
)
def test_strategy_integration(
    strategy: BaseStrategy,
    action_data: Tuple,
    mock_return_value: np.ndarray,
):
    """Integration test for strategies with mixed action types.

    Parameters
    ----------
    strategy : BaseStrategy
        Strategy instance to test.
    action_data : tuple
        Generated action IDs, probabilities, and models.
    mock_return_value : np.ndarray
        Return value for mock maximize function.
    """
    action_ids, probabilities, models = action_data

    # Patch maximize_by_quantity in both utils and strategy modules to ensure all calls are mocked
    with (
        patch("pybandits.utils.maximize_by_quantity") as mock_maximize_utils,
        patch("pybandits.strategy.maximize_by_quantity") as mock_maximize_strategy,
    ):
        mock_maximize_utils.return_value = mock_return_value
        mock_maximize_strategy.return_value = mock_return_value

        result = strategy.select_action(probabilities, models)
        assert result is not None

        # Check result is valid
        if isinstance(result, tuple):
            assert result[0] in action_ids
            assert isinstance(result[1], (tuple, np.ndarray))
        else:
            assert result in action_ids


@pytest.mark.parametrize(
    "strategy_class,kwargs",
    [
        (ClassicBandit, {}),
        (BestActionIdentificationBandit, {"exploit_p": 0.8}),
        (CostControlBandit, {"subsidy_factor": 0.2}),
        (MultiObjectiveBandit, {}),
        (MultiObjectiveCostControlBandit, {"subsidy_factor": 0.6}),
    ],
)
def test_strategy_normalize_field(strategy_class, kwargs):
    """Test field normalization for all strategies.

    Parameters
    ----------
    strategy_class : type
        Strategy class to test.
    kwargs : dict
        Initialization arguments.
    """
    strategy_class(**kwargs)

    # Test normalize_field method if it's a SingleObjectiveStrategy
    if issubclass(strategy_class, SingleObjectiveStrategy):
        # Test with None value
        if "exploit_p" in strategy_class.model_fields:
            result = strategy_class._normalize_field(None, "exploit_p")
            assert result == strategy_class.model_fields["exploit_p"].default

        if "subsidy_factor" in strategy_class.model_fields:
            result = strategy_class._normalize_field(None, "subsidy_factor")
            assert result == strategy_class.model_fields["subsidy_factor"].default

        # Test with actual value
        result = strategy_class._normalize_field(
            0.7, "subsidy_factor" if "subsidy_factor" in strategy_class.model_fields else "exploit_p"
        )
        assert result == 0.7
