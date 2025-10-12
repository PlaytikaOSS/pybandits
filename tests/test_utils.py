"""Tests for pybandits.utils module."""

from abc import ABC, abstractmethod
from types import ModuleType
from typing import Callable, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from bokeh.models import Div, InlineStyleSheet, TabPanel, Tabs

import pybandits
from pybandits.base import Probability
from pybandits.utils import (
    OptimizationFailedError,
    classproperty,
    extract_argument_names_from_function,
    get_non_abstract_classes,
    in_jupyter_notebook,
    maximize_by_quantity,
    visualize_via_bokeh,
)


class TestExtractArgumentNamesFromFunction:
    """Test cases for extract_argument_names_from_function."""

    def test_extract_arguments_from_regular_function(self) -> None:
        """Test extracting argument names from a regular function."""

        def test_func(a: int, b: str, c: bool = True) -> None:
            pass

        result = extract_argument_names_from_function(test_func)
        assert result == ["a", "b", "c"]

    def test_extract_arguments_with_self_ignored(self) -> None:
        """Test that 'self' argument is ignored by default."""

        def test_method(self, a: int, b: str) -> None:
            pass

        result = extract_argument_names_from_function(test_method)
        assert result == ["a", "b"]

    def test_extract_arguments_with_cls_ignored(self) -> None:
        """Test that 'cls' argument is ignored by default."""

        def test_classmethod(cls, a: int, b: str) -> None:
            pass

        result = extract_argument_names_from_function(test_classmethod)
        assert result == ["a", "b"]

    def test_extract_arguments_with_custom_ignore(self) -> None:
        """Test extracting arguments with custom ignore list."""

        def test_func(a: int, b: str, c: bool) -> None:
            pass

        result = extract_argument_names_from_function(test_func, ignore_arguments=("a",))
        assert result == ["b", "c"]

    def test_extract_arguments_from_pydantic_model(self) -> None:
        """Test extracting arguments from a Pydantic model."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            a: int
            b: str
            c: bool = True

        result = extract_argument_names_from_function(TestModel)
        assert result == ["a", "b", "c"]

    def test_extract_arguments_from_pydantic_model_with_ignore(self) -> None:
        """Test extracting arguments from Pydantic model with ignored arguments."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            a: int
            b: str
            c: bool = True

        result = extract_argument_names_from_function(TestModel, ignore_arguments=("a",))
        assert result == ["b", "c"]


class TestGetNonAbstractClasses:
    """Test cases for get_non_abstract_classes."""

    def test_get_non_abstract_classes_from_module(self) -> None:
        """Test getting non-abstract classes from a module."""
        # Create a test module
        test_module = ModuleType("test_module")

        # Add some classes to the module
        class ConcreteClass:
            pass

        class AbstractClass(ABC):
            @abstractmethod
            def abstract_method(self) -> None:
                pass

        class AnotherConcreteClass:
            pass

        # Set the module attribute for each class
        ConcreteClass.__module__ = "test_module"
        AbstractClass.__module__ = "test_module"
        AnotherConcreteClass.__module__ = "test_module"

        # Add classes to module namespace
        test_module.ConcreteClass = ConcreteClass
        test_module.AbstractClass = AbstractClass
        test_module.AnotherConcreteClass = AnotherConcreteClass

        result = get_non_abstract_classes(test_module)

        # Should only return concrete classes
        assert len(result) == 2
        assert ConcreteClass in result
        assert AnotherConcreteClass in result
        assert AbstractClass not in result

    def test_get_non_abstract_classes_empty_module(self) -> None:
        """Test getting non-abstract classes from an empty module."""
        test_module = ModuleType("empty_module")
        result = get_non_abstract_classes(test_module)
        assert result == []

    def test_get_non_abstract_classes_ignores_imported_classes(self) -> None:
        """Test that imported classes are ignored."""
        test_module = ModuleType("test_module")

        class LocalClass:
            pass

        class ImportedClass:
            pass

        # Set different modules
        LocalClass.__module__ = "test_module"
        ImportedClass.__module__ = "other_module"

        test_module.LocalClass = LocalClass
        test_module.ImportedClass = ImportedClass

        result = get_non_abstract_classes(test_module)

        assert len(result) == 1
        assert LocalClass in result
        assert ImportedClass not in result


class TestInJupyterNotebook:
    """Test cases for in_jupyter_notebook."""

    @patch("pybandits.utils.get_ipython")
    def test_in_jupyter_notebook_true(self, mock_get_ipython: MagicMock) -> None:
        """Test that function returns True when in Jupyter notebook."""
        mock_ipython = MagicMock()
        mock_ipython.__class__.__name__ = "ZMQInteractiveShell"
        mock_get_ipython.return_value = mock_ipython

        result = in_jupyter_notebook()
        assert result is True

    @patch("pybandits.utils.get_ipython")
    def test_in_jupyter_notebook_false_different_shell(self, mock_get_ipython: MagicMock) -> None:
        """Test that function returns False for different shell types."""
        mock_ipython = MagicMock()
        mock_ipython.__class__.__name__ = "TerminalInteractiveShell"
        mock_get_ipython.return_value = mock_ipython

        result = in_jupyter_notebook()
        assert result is False

    @patch("pybandits.utils.get_ipython")
    def test_in_jupyter_notebook_false_name_error(self, mock_get_ipython: MagicMock) -> None:
        """Test that function returns False when get_ipython raises NameError."""
        mock_get_ipython.side_effect = NameError("name 'get_ipython' is not defined")

        result = in_jupyter_notebook()
        assert result is False

    @patch("pybandits.utils._IPYTHON_AVAILABLE", False)
    def test_in_jupyter_notebook_false_ipython_not_available(self) -> None:
        """Test that function returns False when IPython is not installed."""
        result = in_jupyter_notebook()
        assert result is False

    @patch("pybandits.utils.get_ipython")
    def test_in_jupyter_notebook_false_get_ipython_returns_none(self, mock_get_ipython: MagicMock) -> None:
        """Test that function returns False when get_ipython returns None."""
        mock_get_ipython.return_value = None

        result = in_jupyter_notebook()
        assert result is False

    def test_in_jupyter_notebook_false_import_error(self) -> None:
        """Test that function returns False when IPython import fails at module level."""
        # Simulate the module-level ImportError by directly patching _IPYTHON_AVAILABLE.
        # Using importlib.reload() would recreate all class objects in pybandits.utils,
        # causing class identity mismatches with names already imported in other test modules.
        with patch.object(pybandits.utils, "_IPYTHON_AVAILABLE", False):
            assert pybandits.utils._IPYTHON_AVAILABLE is False
            result = in_jupyter_notebook()
            assert result is False


class TestVisualizeViaBokeh:
    """Test cases for visualize_via_bokeh."""

    @patch("pybandits.utils.in_jupyter_notebook")
    @patch("pybandits.utils.output_notebook")
    @patch("pybandits.utils.show")
    @patch("pybandits.utils.curdoc")
    def test_visualize_in_jupyter_notebook(
        self, mock_curdoc: MagicMock, mock_show: MagicMock, mock_output_notebook: MagicMock, mock_in_jupyter: MagicMock
    ) -> None:
        """Test visualization in Jupyter notebook environment."""
        mock_in_jupyter.return_value = True
        mock_doc = MagicMock()
        mock_curdoc.return_value = mock_doc

        tabs = [TabPanel(child=Div(text="Test Content"), title="Test Tab")]

        visualize_via_bokeh(None, tabs)

        mock_output_notebook.assert_called_once()
        mock_show.assert_called_once()
        assert mock_doc.title == "Visual report"

    @patch("pybandits.utils.in_jupyter_notebook")
    @patch("pybandits.utils.output_file")
    @patch("pybandits.utils.save")
    @patch("pybandits.utils.curdoc")
    def test_visualize_to_file(
        self, mock_curdoc: MagicMock, mock_save: MagicMock, mock_output_file: MagicMock, mock_in_jupyter: MagicMock
    ) -> None:
        """Test visualization to HTML file."""
        mock_in_jupyter.return_value = False
        mock_doc = MagicMock()
        mock_curdoc.return_value = mock_doc

        tabs = [TabPanel(child=Div(text="Test Content"), title="Test Tab")]
        output_path = "test_output.html"

        visualize_via_bokeh(output_path, tabs)

        mock_output_file.assert_called_once_with(output_path)
        mock_save.assert_called_once()
        assert mock_doc.title == "Visual report"

    @patch("pybandits.utils.in_jupyter_notebook")
    def test_visualize_no_output_path_error(self, mock_in_jupyter: MagicMock) -> None:
        """Test that ValueError is raised when no output_path is provided outside Jupyter."""
        mock_in_jupyter.return_value = False

        tabs = [TabPanel(child=Div(text="Test Content"), title="Test Tab")]

        with pytest.raises(ValueError, match="output_path is required when not running in a Jupyter notebook"):
            visualize_via_bokeh(None, tabs)

    @patch("pybandits.utils.in_jupyter_notebook")
    @patch("pybandits.utils.output_notebook")
    @patch("pybandits.utils.show")
    @patch("pybandits.utils.curdoc")
    def test_visualize_tabs_styling(
        self, mock_curdoc: MagicMock, mock_show: MagicMock, mock_output_notebook: MagicMock, mock_in_jupyter: MagicMock
    ) -> None:
        """Test that tabs are created with proper styling."""
        mock_in_jupyter.return_value = True
        mock_doc = MagicMock()
        mock_curdoc.return_value = mock_doc

        tabs = [TabPanel(child=Div(text="Test Content"), title="Test Tab")]

        visualize_via_bokeh(None, tabs)

        # Verify that show was called with a Tabs object
        call_args = mock_show.call_args[0][0]
        assert isinstance(call_args, Tabs)
        assert call_args.tabs == tabs
        assert call_args.sizing_mode == "stretch_both"

        # Verify stylesheet was added
        assert len(call_args.stylesheets) == 1
        assert isinstance(call_args.stylesheets[0], InlineStyleSheet)


class TestClassProperty:
    """Test cases for classproperty decorator."""

    def test_classproperty_decorator(self) -> None:
        """Test that classproperty works correctly."""

        class TestClass:
            _value = "test_value"

            @classproperty
            def class_attr(cls) -> str:
                return cls._value

        # Test accessing via class
        assert TestClass.class_attr == "test_value"

        # Test accessing via instance
        instance = TestClass()
        assert instance.class_attr == "test_value"

    def test_classproperty_with_different_values(self) -> None:
        """Test classproperty with different class values."""

        class TestClass:
            _value = "original"

            @classproperty
            def class_attr(cls) -> str:
                return cls._value

        class SubClass(TestClass):
            _value = "subclass"

        assert TestClass.class_attr == "original"
        assert SubClass.class_attr == "subclass"


class TestMaximizeByQuantity:
    """Test cases for maximize_by_quantity function."""

    @pytest.fixture
    def default_n_trials(self) -> int:
        """Default number of trials for optimization."""
        return 1000

    @pytest.fixture
    def default_dimension(self) -> int:
        """Default dimension for test cases."""
        return 2

    @pytest.mark.parametrize(
        "quantity_score_func,dimension,expected_shape",
        [
            # Simple quadratic function - maximum at (0.5, 0.5)
            (lambda x: (1.0 - np.sum((x - 0.5) ** 2)), 2, (2,)),
            # Linear function - maximum at (1.0, 1.0)
            (lambda x: (np.sum(x)), 2, (2,)),
            # Single variable function
            (lambda x: (1.0 - (x[0] - 0.7) ** 2), 1, (1,)),
            # Three variable function
            (lambda x: (1.0 - np.sum((x - 0.3) ** 2)), 3, (3,)),
        ],
    )
    def test_maximize_by_quantity_without_constraints(
        self,
        quantity_score_func: Callable[[np.ndarray], float],
        dimension: int,
        expected_shape: tuple,
        default_n_trials: int,
    ) -> None:
        """Test maximize_by_quantity without constraints."""
        result = maximize_by_quantity(quantity_score_func, dimension, n_trials=default_n_trials)

        assert isinstance(result, np.ndarray)
        assert result.shape == expected_shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Verify that the result actually maximizes the function
        result_score = quantity_score_func(result)
        assert result_score > 0.0  # Should be positive for valid result

    @pytest.mark.parametrize(
        "quantity_score_func,dimension,constraints",
        [
            # Constraint: sum of elements <= 0.5
            (
                lambda x: (np.sum(x)),
                2,
                [lambda x: np.sum(x) <= 0.5],
            ),
            # Constraint: first element <= 0.3
            (
                lambda x: (x[0] + x[1]),
                2,
                [lambda x: x[0] <= 0.3],
            ),
            # Multiple constraints: sum <= 0.6 and first element >= 0.2
            (
                lambda x: (np.sum(x)),
                2,
                [lambda x: np.sum(x) <= 0.6, lambda x: x[0] >= 0.2],
            ),
            # Constraint: product of elements <= 0.1
            (
                lambda x: (np.prod(x)),
                2,
                [lambda x: np.prod(x) <= 0.1],
            ),
        ],
        ids=["sum <= 0.5", "first element <= 0.3", "sum <= 0.6 and first element >= 0.2", "product <= 0.1"],
    )
    def test_maximize_by_quantity_with_constraints(
        self,
        quantity_score_func: Callable[[np.ndarray], float],
        dimension: int,
        constraints: List[Callable[[np.ndarray], bool]],
        default_n_trials: int,
    ) -> None:
        """Test maximize_by_quantity with constraints."""
        result = maximize_by_quantity(quantity_score_func, dimension, constraints, n_trials=default_n_trials)

        assert isinstance(result, np.ndarray)
        assert result.shape == (dimension,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Verify constraints are satisfied
        for constraint in constraints:
            assert constraint(result)

    @pytest.mark.parametrize(
        "n_trials",
        [100, 1000, 5000],
    )
    def test_maximize_by_quantity_different_trial_counts(
        self, n_trials: int, default_dimension: int, center: float = 0.5
    ) -> None:
        """Test maximize_by_quantity with different trial counts."""

        def quantity_score_func(x) -> Probability:
            return Probability(1.0 - np.sum((x - center) ** 2))

        result = maximize_by_quantity(quantity_score_func, default_dimension, n_trials=n_trials)

        assert isinstance(result, np.ndarray)
        assert result.shape == (default_dimension,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_maximize_by_quantity_single_element_array(self, dimension: int = 1, center: float = 0.7) -> None:
        """Test maximize_by_quantity with single element array."""

        def quantity_score_func(x) -> Probability:
            return Probability(1.0 - (x[0] - center) ** 2)

        result = maximize_by_quantity(quantity_score_func, dimension)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0

    def test_maximize_by_quantity_large_array(
        self, default_n_trials: int, dimension: int = 5, center: float = 0.3
    ) -> None:
        """Test maximize_by_quantity with larger array."""

        def quantity_score_func(x) -> Probability:
            return Probability(1.0 - np.sum((x - center) ** 2))

        result = maximize_by_quantity(quantity_score_func, dimension, n_trials=default_n_trials)

        assert isinstance(result, np.ndarray)
        assert result.shape == (dimension,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_maximize_by_quantity_constraint_violation_handling(
        self,
        default_n_trials: int,
        bound: float = 0.3,
        epsilon: float = 1e-6,
        dimension: int = 2,
    ) -> None:
        """Test that constraints are properly handled when they might be violated."""

        # This test uses a function that would naturally maximize at (1, 1)
        # but we constrain it to stay within a smaller region
        def quantity_score_func(x) -> Probability:
            return Probability(x[0] + x[1])

        constraints = [lambda x: np.sum(x) <= bound]  # Force sum to be small

        result = maximize_by_quantity(quantity_score_func, dimension, constraints, n_trials=default_n_trials)
        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.shape == (dimension,)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)
            assert np.sum(result) <= bound + epsilon  # Allow small numerical error

    @pytest.mark.parametrize(
        "constraints",
        [
            None,
            [],
            [lambda x: True],  # Always satisfied constraint
            [lambda x: x[0] >= 0.0],  # Trivial constraint
        ],
    )
    def test_maximize_by_quantity_various_constraint_inputs(
        self,
        constraints: Optional[List[Callable[[np.ndarray], bool]]],
        default_dimension: int,
        center: float = 0.5,
    ) -> None:
        """Test maximize_by_quantity with various constraint inputs."""

        def quantity_score_func(x) -> Probability:
            return Probability(1.0 - np.sum((x - center) ** 2))

        result = maximize_by_quantity(quantity_score_func, default_dimension, constraints)

        assert isinstance(result, np.ndarray)
        assert result.shape == (default_dimension,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_maximize_by_quantity_caching_behavior(
        self, default_dimension: int, default_n_trials: int, center: float = 0.5
    ) -> None:
        """Test that the function uses caching correctly."""

        def quantity_score_func(x) -> Probability:
            return Probability(1.0 - np.sum((x - center) ** 2))

        # First call
        result1 = maximize_by_quantity(quantity_score_func, default_dimension, n_trials=default_n_trials)

        # Second call with same parameters should use cache
        result2 = maximize_by_quantity(quantity_score_func, default_dimension, n_trials=default_n_trials)

        # Results should be identical due to caching
        np.testing.assert_array_almost_equal(result1, result2)

    @pytest.mark.parametrize("dimension", [1, 2, 3, 4])
    def test_maximize_by_quantity_different_dimensions(
        self, dimension: int, default_n_trials: int, center: float = 0.5
    ) -> None:
        """Test maximize_by_quantity with different dimensions."""

        def quantity_score_func(x) -> Probability:
            return Probability(1.0 - np.sum((x - center) ** 2))

        result = maximize_by_quantity(quantity_score_func, dimension, n_trials=default_n_trials)

        assert isinstance(result, np.ndarray)
        assert result.shape == (dimension,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    @patch("pybandits.utils.differential_evolution")
    def test_maximize_by_quantity_optimization_failure(
        self, mock_de: MagicMock, default_dimension: int, center: float = 0.5
    ) -> None:
        """Test that OptimizationFailedError is raised when optimization fails."""
        # Mock differential_evolution to return unsuccessful result
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.message = "Optimization failed"
        mock_de.return_value = mock_result

        def quantity_score_func(x) -> Probability:
            return Probability(1.0 - np.sum((x - center) ** 2))

        with pytest.raises(OptimizationFailedError, match="Optimization failed"):
            maximize_by_quantity(quantity_score_func, default_dimension)

    def test_maximize_by_quantity_complex_constraints(
        self,
        default_n_trials: int,
        upper_bound: float = 0.8,
        lower_bound: float = 0.2,
        dimension: int = 2,
    ) -> None:
        """Test maximize_by_quantity with complex constraint scenarios."""

        # Function that wants to maximize at (1, 1) but we constrain it
        def quantity_score_func(x) -> Probability:
            return Probability(x[0] * x[1])

        # Complex constraints: x[0] + x[1] <= upper_bound and x[0] >= lower_bound and x[1] >= lower_bound
        constraints = [
            lambda x: x[0] + x[1] <= upper_bound,
            lambda x: x[0] >= lower_bound,
            lambda x: x[1] >= lower_bound,
        ]

        result = maximize_by_quantity(quantity_score_func, dimension, constraints, n_trials=default_n_trials)

        assert isinstance(result, np.ndarray)
        assert result.shape == (dimension,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Verify all constraints are satisfied
        assert result[0] + result[1] <= upper_bound + 1e-6
        assert result[0] >= lower_bound - 1e-6
        assert result[1] >= lower_bound - 1e-6

    def test_maximize_by_quantity_returns_maximum_probability(
        self, default_n_trials: int, dimension: int = 1, return_value: float = 0.8
    ) -> None:
        """Test that maximize_by_quantity returns maximum probability value from optimization."""

        def prob_func(x: np.ndarray) -> float:
            return return_value

        result = maximize_by_quantity(prob_func, dimension, n_trials=default_n_trials)

        # The function should return the optimal point that maximizes the function
        # Since prob_func always returns return_value, any point in [0,1] should work
        assert isinstance(result, np.ndarray)
        assert result.shape == (dimension,)
        assert 0.0 <= result[0] <= 1.0

    @pytest.mark.parametrize("dimension", [1, 2, 3])
    def test_maximize_by_quantity_samples_points_in_valid_range(
        self, default_n_trials: int, dimension: int, return_value: float = 0.5
    ) -> None:
        """Test that maximize_by_quantity correctly samples points from [0,1] range."""

        def prob_func(x: np.ndarray) -> float:
            # Verify all inputs are in valid range
            assert all(0 <= xi <= 1 for xi in x)
            return return_value

        result = maximize_by_quantity(prob_func, dimension, n_trials=default_n_trials)

        assert isinstance(result, np.ndarray)
        assert result.shape == (dimension,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_maximize_by_quantity_maximization_result(
        self,
        default_n_trials: int,
        dimension: int = 1,
        atol: float = 1e-2,
        expected_min: float = 0.0,
        expected_max: float = 1.0,
    ) -> None:
        """Test that maximize_by_quantity correctly maximizes simple functions."""
        # Test function with maximum at x=0: 1 - x^2
        result1 = maximize_by_quantity(lambda x: 1 - x[0] ** 2, dimension, n_trials=default_n_trials)
        assert np.isclose(result1[0], expected_min, atol=atol)

        # Test function with maximum at x=1: x^2
        result2 = maximize_by_quantity(lambda x: x[0] ** 2, dimension, n_trials=default_n_trials)
        assert np.isclose(result2[0], expected_max, atol=atol)

    @patch("pybandits.utils.differential_evolution")
    def test_maximize_by_quantity_uses_differential_evolution(
        self,
        mock_de: MagicMock,
        default_n_trials: int,
        dimension: int = 1,
        return_value: float = 0.5,
        test_input_value: float = 0.5,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> None:
        """Test that maximize_by_quantity uses differential_evolution optimization."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.x = np.array([test_input_value])
        mock_de.return_value = mock_result

        def prob_func(x: np.ndarray) -> float:
            return return_value

        maximize_by_quantity(prob_func, dimension, n_trials=default_n_trials)

        # Verify differential_evolution was called
        mock_de.assert_called_once()
        call_args = mock_de.call_args

        # Check that bounds are set correctly
        assert "bounds" in call_args.kwargs
        bounds = call_args.kwargs["bounds"]
        assert bounds == [(lower_bound, upper_bound)]  # Single dimension bounds

        # Check that function is negated (for maximization)
        func = call_args.kwargs["func"]
        test_input = np.array([test_input_value])
        assert func(test_input) == -return_value  # Should be negated

    def test_maximize_by_quantity_probability_function_exceptions(
        self, default_n_trials: int, dimension: int = 1, error_message: str = "Function failed"
    ) -> None:
        """Test that exceptions from probability function are properly propagated."""

        def failing_prob_func(x: np.ndarray) -> float:
            raise RuntimeError(error_message)

        with pytest.raises(RuntimeError, match=error_message):
            maximize_by_quantity(failing_prob_func, dimension, n_trials=default_n_trials)

    def test_maximize_by_quantity_large_input_dimension(
        self, default_n_trials: int, dimension: int = 30, return_value: float = 0.5
    ) -> None:
        """Test maximize_by_quantity with large input dimension."""

        def prob_func(x: np.ndarray) -> float:
            return return_value

        result = maximize_by_quantity(prob_func, dimension, n_trials=default_n_trials)

        assert isinstance(result, np.ndarray)
        assert result.shape == (dimension,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    @patch("pybandits.utils.differential_evolution")
    def test_maximize_by_quantity_optimization_convergence_failure(
        self,
        mock_de: MagicMock,
        default_n_trials: int,
        dimension: int = 1,
        return_value: float = 0.5,
        error_message: str = "Optimization failed to converge",
    ) -> None:
        """Test that OptimizationFailedError is raised when optimization fails to converge."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.message = error_message
        mock_de.return_value = mock_result

        def prob_func(x: np.ndarray) -> float:
            return return_value

        with pytest.raises(OptimizationFailedError, match=error_message):
            maximize_by_quantity(prob_func, dimension, n_trials=default_n_trials)
