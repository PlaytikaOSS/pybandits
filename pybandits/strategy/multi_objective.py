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

from abc import ABC
from typing import Callable, ClassVar, Dict, Generator, List, Optional, Type, Union

import numpy as np
from loguru import logger
from pydantic import PrivateAttr, validate_call

from pybandits.base import ActionId, UnifiedActionId
from pybandits.base_model import BaseModel
from pybandits.quantitative_model import QuantitativeModel
from pybandits.strategy.base import BaseStrategy, CostControlStrategy, SingleObjectiveStrategy
from pybandits.strategy.single_objective import ClassicBandit, CostControlBandit


class MultiObjectiveStrategy(BaseStrategy, ABC):
    """
    Abstract strategy for multi-objective multi-armed bandits.

    This class handles bandits where each action has multiple reward objectives.
    It selects actions from the Pareto front - the set of non-dominated actions
    where no other action is better in all objectives.
    """

    # Class variable to define how to select the best action for each objective
    objective_selector_class: ClassVar[Type[SingleObjectiveStrategy]]
    _objective_selector: SingleObjectiveStrategy = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._objective_selector = self.objective_selector_class(**data)

    @validate_call
    def select_action(
        self,
        p: Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]],
        actions: Dict[ActionId, BaseModel],
    ) -> UnifiedActionId:
        """
        Select an action from the Pareto front.

        This method finds all Pareto-optimal actions and randomly selects one,
        giving equal probability to each non-dominated action.

        Parameters
        ----------
        p : Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]]
            Dictionary mapping action IDs to either:
            - List[float]: Fixed reward vector for multiple objectives
            - Callable: Function that computes reward vector given quantity
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their associated models.

        Returns
        -------
        UnifiedActionId
            A randomly selected action from the Pareto front.
        """
        pareto_front = self._get_pareto_front(p=p, actions=actions)
        return np.random.choice(pareto_front)

    def _get_feasible_solutions(
        self,
        p: Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]],
        actions: Dict[ActionId, BaseModel],
    ) -> Dict[UnifiedActionId, List[float]]:
        """
        Get feasible solutions for each objective.

        Applies logic independently to each objective, finding actions that meet the selection that objective.

        Parameters
        ----------
        p : Dict[ActionId, List[float]]
            Dictionary mapping action IDs to reward vectors.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models.

        Returns
        -------
        Dict[UnifiedActionId, List[float]]
            Feasible actions considering logic for each objective.
        """
        action_id = list(p.keys())[0]
        if isinstance(action_id, tuple):
            action_id = action_id[0]
        n_objectives = len(actions[action_id].models)
        feasible_solutions = {}
        # Separate discrete and quantitative actions
        discrete_actions = {aid: prob_or_func for aid, prob_or_func in p.items() if not callable(prob_or_func)}
        quantitative_actions = {aid: prob_or_func for aid, prob_or_func in p.items() if callable(prob_or_func)}

        # For discrete actions, add directly (they already have full reward vectors)
        feasible_solutions.update(discrete_actions)

        # For quantitative actions, refine per objective
        if quantitative_actions:
            for i in range(n_objectives):
                # Fix closure bug: create a new function that captures i by value
                def make_objective_extractor(obj_idx):
                    return lambda x: x[obj_idx]

                objective_p = {action_id: make_objective_extractor(i) for action_id in quantitative_actions.keys()}
                objective_actions = {
                    action_id: actions[action_id].models[i] for action_id in quantitative_actions.keys()
                }

                refined = self._objective_selector.refine_p(objective_p, objective_actions, None)

                # Build multi-objective vectors from per-objective results
                for unified_action_id in refined.keys():
                    if unified_action_id not in feasible_solutions:
                        feasible_solutions[unified_action_id] = quantitative_actions[unified_action_id[0]](
                            unified_action_id[1]
                        )

        return feasible_solutions

    def _get_exact_pareto_front(
        self, p: Dict[UnifiedActionId, List[float]], actions: Dict[ActionId, BaseModel]
    ) -> List[UnifiedActionId]:
        """
        Compute the exact Pareto front for discrete action sets.

        An action is Pareto-optimal if no other action dominates it (i.e., is
        better or equal in all objectives and strictly better in at least one).

        Parameters
        ----------
        p : Dict[UnifiedActionId, List[float]]
            Dictionary mapping unified action IDs to their reward vectors.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models.

        Returns
        -------
        List[UnifiedActionId]
            List of Pareto-optimal action IDs.
        """
        feasible_solutions = self._get_feasible_solutions(p, actions)
        # store non dominated actions
        pareto_front = []

        for this_action in feasible_solutions.keys():
            is_pareto = True  # we assume that action is Pareto Optimal until proven otherwise
            other_actions = [a for a in feasible_solutions.keys() if a != this_action]

            for other_action in other_actions:
                # check if this_action is not dominated by other_action based on
                # multiple objectives reward prob vectors
                is_dominated = not (
                    # an action cannot be dominated by an identical one
                    (feasible_solutions[this_action] == feasible_solutions[other_action])
                    # otherwise, apply the classical definition
                    or any(
                        feasible_solutions[this_action][i] > feasible_solutions[other_action][i]
                        for i in range(len(feasible_solutions[this_action]))
                    )
                )

                if is_dominated:
                    # this_action dominated by at least one other_action,
                    # this_action is not pareto optimal
                    is_pareto = False
                    break

            if is_pareto:
                # this_action is pareto optimal
                pareto_front.append(this_action)

        return pareto_front

    def _get_approximate_pareto_front(
        self,
        p: Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]],
        actions: Dict[ActionId, BaseModel],
        n_divisions: int = 10,
    ) -> List[UnifiedActionId]:
        """
        Approximate the Pareto front for continuous/quantitative actions.

        Uses the Normal Constraint method with Das-Dennis weight generation to
        systematically sample the Pareto front for quantitative actions.

        Parameters
        ----------
        p : Dict[ActionId, Union[List[float], Callable[[np.ndarray], List[float]]]]
            Dictionary mapping action IDs to reward vectors or functions.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models.
        n_divisions : int, default=10
            Number of divisions for weight vector generation. Higher values
            provide better approximation but increase computation.

        Returns
        -------
        List[UnifiedActionId]
            List of approximately Pareto-optimal actions.
        """
        if not p:
            return []

        approximate_p = {}
        n_objectives = len(actions[list(p.keys())[0]].models)

        for action_id, prob_or_func in p.items():
            if callable(prob_or_func):
                # Quantitative action - find Pareto optimal input points
                pareto_input_points = self._find_pareto_front_normal_constraint(
                    prob_or_func, actions[action_id].dimension, n_objectives, n_divisions, actions[action_id]
                )
                approximate_p.update(
                    {(action_id, tuple(input_point)): prob_or_func(input_point) for input_point in pareto_input_points}
                )
            else:
                # Standard action with fixed reward vector
                approximate_p[action_id] = prob_or_func

        return self._get_exact_pareto_front(approximate_p, actions)

    @validate_call
    def _find_pareto_front_normal_constraint(
        self,
        func: Callable[[np.ndarray], List[float]],
        input_dim: int,
        n_objectives: int,
        n_divisions: int,
        model: BaseModel,
    ) -> List[np.ndarray]:
        """
        Find Pareto front using Normal Constraint method with Das-Dennis weight generation for a single function.

        This method systematically explores the Pareto front by solving constrained
        optimization problems with different weight vectors.

        Parameters
        ----------
        func : Callable[[np.ndarray], List[float]]
            Function mapping quantity vectors to reward vectors.
        input_dim : int
            Dimension of the input quantity vector.
        n_objectives : int
            Number of reward objectives.
        n_divisions : int
            Number of divisions for weight generation (controls approximation quality).
        model : BaseModel
            The model for this quantitative action.

        Returns
        -------
        List[np.ndarray]
            List of Pareto-optimal quantity vectors.

        References
        ----------
        The normalized normal constraint method for generating the Pareto frontier (Messac et al., 2003)
        https://ieeexplore.ieee.org/document/938649
        """
        # Step 1: Find anchor points using optimization for each objective
        anchor_points = [
            self._objective_selector.verify_and_select_from_quantitative_action(
                lambda x: func(x)[i], model.models[i], None
            )
            for i in range(n_objectives)
        ]
        anchor_rewards = [func(anchor_point) for anchor_point in anchor_points]

        anchor_matrix = np.array(anchor_rewards)  # n_objectives x n_objectives
        anchor_points = np.array(anchor_points)  # n_objectives x input_dim

        # Step 2: Generate Das-Dennis weight vectors
        weight_vectors = self._das_dennis_weights(n_objectives, n_divisions)

        # Step 3: For each weight vector, solve NC subproblem
        nc_solutions = set(tuple(anchor_point) for anchor_point in anchor_points)
        utopia = np.max(anchor_matrix, axis=0)  # Ideal point

        for weight in weight_vectors:
            solution = self._solve_nc_subproblem(func, anchor_matrix, utopia, weight, model)
            if solution is not None:
                nc_solutions.add(tuple(solution))

        return list(nc_solutions)

    @staticmethod
    def _das_dennis_weights(n_objectives: int, n_divisions: int) -> np.ndarray:
        """
        Generate Das-Dennis weight vectors for systematic Pareto front sampling.

        Creates uniformly distributed weight vectors on the unit simplex using
        the Das-Dennis method, which provides systematic coverage of the
        objective space.

        Parameters
        ----------
        n_objectives : int
            Number of objectives/dimensions.
        n_divisions : int
            Number of divisions per dimension. Total weights generated is
            approximately (n_divisions + n_objectives - 1)! / (n_divisions! * (n_objectives - 1)!).

        Returns
        -------
        np.ndarray
            Array of shape (n_weights, n_objectives) containing weight vectors.
        """

        def generate_recursive(
            n_obj: int, n_div: int, current_weight: List[int], depth: int = 0
        ) -> Generator[np.ndarray, None, None]:
            """
            Recursively generate weight combinations for Das-Dennis method.

            Parameters
            ----------
            n_obj : int
                Number of objectives.
            n_div : int
                Remaining divisions to allocate.
            current_weight : List[int]
                Current partial weight vector being built.
            depth : int
                Current recursion depth (objective index).

            Yields
            ------
            np.ndarray
                Normalized weight vectors summing to 1.
            """
            if depth == n_obj - 1:
                current_weight.append(n_div)
                yield np.array(current_weight) / n_divisions
                current_weight.pop()
                return

            for i in range(n_div + 1):
                current_weight.append(i)
                yield from generate_recursive(n_obj, n_div - i, current_weight, depth + 1)
                current_weight.pop()

        weights = list(generate_recursive(n_objectives, n_divisions, []))
        return np.array(weights)

    def _solve_nc_subproblem(
        self,
        func: Callable,
        anchor_matrix: np.ndarray,
        utopia: np.ndarray,
        weight: np.ndarray,
        model: BaseModel,
        epsilon: float = 1e-10,
    ) -> Optional[np.ndarray]:
        """
        Solve a single Normal Constraint optimization subproblem.

        Maximizes a weighted objective while constraining other objectives to lie
        on the "reference point side" of hyperplanes through anchor points.

        Parameters
        ----------
        func : Callable
            The multi-objective function to optimize.
        anchor_matrix : np.ndarray
            Matrix of anchor points (extreme points for each objective).
        utopia : np.ndarray
            The utopia point (ideal but typically unachievable point).
        weight : np.ndarray
            Weight vector determining the reference point and primary objective.
        model : BaseModel
            The model for constraint evaluation.
        epsilon : float, default=1e-10
            Numerical tolerance for constraint satisfaction.

        Returns
        -------
        Optional[np.ndarray]
            Optimal solution if found and feasible, None otherwise.
        """
        n_objectives = len(weight)
        primary_obj = np.argmax(weight)

        # Step #1: Create the utopia-based coordinate system
        # Transform the problem so utopia is at origin
        transformed_anchors = anchor_matrix - utopia  # Anchors relative to utopia

        # Step #2: Find reference point using weight vector from utopia
        # This is where the weight ray from utopia intersects the anchor hyperplane
        reference_point_transformed = self._find_utopia_reference_point(transformed_anchors, weight, epsilon)
        reference_point = reference_point_transformed + utopia  # Back to original coordinates

        # Step #3: Create Normal Constraint boundaries using utopia geometry
        constraint_normals = []
        constraint_intercepts = []

        for i in range(n_objectives):
            if i != primary_obj:
                # Normal vector points from anchor_i towards utopia
                # This creates the "feasible cone" emanating from utopia
                normal_direction = reference_point - anchor_matrix[i]

                # The constraint hyperplane passes through anchor_i with this normal
                if np.linalg.norm(normal_direction) > epsilon:
                    normal = normal_direction / np.linalg.norm(normal_direction)
                    intercept = np.dot(normal, anchor_matrix[i])

                    constraint_normals.append(normal)
                    constraint_intercepts.append(intercept)

        def reference_based_constraints(x: np.ndarray) -> bool:
            """
            Check if a point satisfies Normal Constraint boundaries.

            Verifies that the function value at x lies on the correct side of all
            constraint hyperplanes defined by the anchor points and reference point.

            Parameters
            ----------
            x : np.ndarray
                Input point to evaluate.

            Returns
            -------
            bool
                True if all constraints are satisfied, False otherwise.
            """
            rewards = np.array(func(x))

            for normal, intercept in zip(constraint_normals, constraint_intercepts):
                # Constraint: normal · f(x) >= intercept
                # Geometric meaning: f(x) is on the reference point side of the boundary
                constraint_value = np.dot(normal, rewards) - intercept

                if constraint_value < -epsilon:  # Tolerance for numerical errors
                    return False
            return True

        def objective_function(x: np.ndarray) -> float:
            """
            Extract the primary objective value for maximization.

            Parameters
            ----------
            x : np.ndarray
                Input point to evaluate.

            Returns
            -------
            float
                Value of the primary objective at x.
            """
            return func(x)[primary_obj]

        # Solve the constrained optimization
        try:
            solution = self._objective_selector.verify_and_select_from_quantitative_action(
                objective_function, model.models[primary_obj], reference_based_constraints
            )

            if reference_based_constraints(solution):
                return solution
            else:
                return None

        except Exception as e:
            logger.error(f"NC subproblem failed: {e}")
            return None

    @classmethod
    def _find_utopia_reference_point(
        cls, transformed_anchors: np.ndarray, weight: np.ndarray, epsilon: float
    ) -> np.ndarray:
        """
        Find the reference point for Normal Constraint method.

        Computes where a ray from the utopia point in the direction of the weight
        vector intersects the hyperplane defined by the anchor points.

        Parameters
        ----------
        transformed_anchors : np.ndarray
            Anchor points transformed relative to utopia point.
        weight : np.ndarray
            Direction vector from utopia point.
        epsilon : float
            Numerical tolerance for degeneracy detection.

        Returns
        -------
        np.ndarray
            The reference point in the transformed coordinate system.
        """

        # ray-hyperplane intersection
        anchor_center = np.mean(transformed_anchors, axis=0)
        anchor_vectors = transformed_anchors - anchor_center

        try:
            # Find hyperplane normal
            U, _, _ = np.linalg.svd(anchor_vectors.T, full_matrices=True)
            hyperplane_normal = U[:, -1]

            # Ray intersection
            numerator = np.dot(hyperplane_normal, anchor_center)
            denominator = np.dot(hyperplane_normal, weight)

            if abs(denominator) > epsilon:
                t = numerator / denominator
                intersection = t * weight
                return intersection
            else:
                return np.dot(weight, transformed_anchors)

        except np.linalg.LinAlgError:
            return np.dot(weight, transformed_anchors)

    def _get_pareto_front(
        self,
        p: Dict[ActionId, Union[List[float], List[Callable[[np.ndarray], float]]]],
        actions: Dict[ActionId, BaseModel],
    ) -> List[UnifiedActionId]:
        """
        Compute the Pareto front, using exact or approximate methods as appropriate.

        Automatically selects between exact computation (for discrete actions) and
        approximation (when quantitative actions are present).

        Parameters
        ----------
        p : Dict[ActionId, Union[List[float], List[Callable[[np.ndarray], float]]]]
            Dictionary mapping action IDs to reward vectors or functions.
        actions : Dict[ActionId, BaseModel]
            Dictionary mapping action IDs to their models.

        Returns
        -------
        List[UnifiedActionId]
            List of Pareto-optimal actions.
        """
        includes_quantitative_actions = any(isinstance(actions[a], QuantitativeModel) for a in p.keys())
        return (
            self._get_approximate_pareto_front(p, actions)
            if includes_quantitative_actions
            else self._get_exact_pareto_front(p, actions)
        )


class MultiObjectiveBandit(MultiObjectiveStrategy):
    """
    Multi-objective Thompson Sampling strategy for multi-armed bandits.

    This strategy handles vector-valued rewards where each action produces multiple
    reward outcomes. Actions are selected from the Pareto front - the set of
    non-dominated actions where no other action is superior in all objectives.

    The strategy uses Thompson Sampling for exploration by sampling from posterior
    distributions and then selecting uniformly from the resulting Pareto front.



    References
    ----------
    Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem (Yahyaa and Manderick, 2015)
    https://www.researchgate.net/publication/272823659_Thompson_Sampling_for_Multi-Objective_Multi-Armed_Bandits_Problem
    """

    # Use ClassicBandit's selection strategy for finding extreme points
    objective_selector_class: ClassVar[Type[SingleObjectiveStrategy]] = ClassicBandit


class MultiObjectiveCostControlBandit(MultiObjectiveStrategy, CostControlStrategy):
    """
    Multi-objective strategy with cost control for multi-armed bandits.

    Combines multi-objective optimization with cost awareness. For each objective,
    identifies actions within a tolerance of the best reward, then considers only
    the lowest-cost actions from these feasible sets when computing the Pareto front.

    This strategy is useful when actions have both multiple reward objectives and
    associated costs, requiring a balance between Pareto-optimality and cost efficiency.



    """

    # Use CostControlBandit's selection strategy for finding extreme points
    objective_selector_class: ClassVar[Type[SingleObjectiveStrategy]] = CostControlBandit
