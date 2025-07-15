"""Tests for pybandits.utils module."""

from abc import ABC, abstractmethod
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from bokeh.models import Div, InlineStyleSheet, TabPanel, Tabs

from pybandits.utils import (
    classproperty,
    extract_argument_names_from_function,
    get_non_abstract_classes,
    in_jupyter_notebook,
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
