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

"""
This snippet is a compatibility layer for pydantic v1 and v2.
"""

from typing import Any, Callable, Dict, Literal, Optional, Union
from warnings import warn

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    ValidationError,
    confloat,
    conint,
    constr,
)
from pydantic.version import VERSION as _VERSION

# Define the pydantic versions
PYDANTIC_VERSION_1 = "1"
PYDANTIC_VERSION_2 = "2"


def _get_major_pydantic_version():
    """
    Get the major version of pydantic.

    Returns
    -------
    major_version : str
        The major version of pydantic.
    """
    try:
        major_version = _VERSION.split(".")[0]
        return major_version
    except Exception as e:
        raise ValueError(f"Error getting Pydantic version: {e}")


pydantic_version = _get_major_pydantic_version()

if pydantic_version == PYDANTIC_VERSION_1:
    from pydantic import root_validator, validate_arguments, validator
    from pydantic.typing import AnyCallable as _AnyCallable

    # In pydantic v1, the field, model, and call validators are defined differently
    def field_validator(
        field: str,
        /,
        *fields: str,
        mode: Literal["after", "before"] = "after",
        check_fields: bool = True,
    ):
        """
        Decorate methods on the class indicating that they should be used to validate fields.

        Example usage:
        ```py
        from typing import Any

        from pybandits.pydantic_version_compatibility import (
            BaseModel,
            ValidationError,
            field_validator,
        )

        class Model(BaseModel):
            a: str

            @field_validator('a')
            @classmethod
            def ensure_foobar(cls, v: Any):
                if 'foobar' not in v:
                    raise ValueError('"foobar" not found in a')
                return v

        print(repr(Model(a='this is foobar good')))
        #> Model(a='this is foobar good')

        try:
            Model(a='snap')
        except ValidationError as exc_info:
            print(exc_info)
            '''
            1 validation error for Model
            a
              Value error, "foobar" not found in a [type=value_error, input_value='snap', input_type=str]
            '''
        ```

        For more in depth examples, see [Field Validators](../concepts/validators.md#field-validators).

        Args:
            field: The first field the `field_validator` should be called on; this is separate
                from `fields` to ensure an error is raised if you don't pass at least one.
            *fields: Additional field(s) the `field_validator` should be called on.
            mode: Specifies whether to validate the fields before or after validation.
            check_fields: Whether to check that the fields actually exist on the model.

        Returns:
            A decorator that can be used to decorate a function to be used as a field_validator.

        Raises:
            PydanticUserError:
                - If `@field_validator` is used bare (with no fields).
                - If the args passed to `@field_validator` as fields are not strings.
                - If `@field_validator` applied to instance methods.
        """
        pre = mode == "before"
        fields = field, *fields
        return validator(*fields, pre=pre, check_fields=check_fields)

    def model_validator(
        *,
        mode: Literal["before", "after"],
    ) -> Any:
        """
        Decorate model methods for validation purposes.

        Example usage:
        ```py
        from typing_extensions import Self

        from pybandits.pydantic_version_compatibility import BaseModel, ValidationError, model_validator

        class Square(BaseModel):
            width: float
            height: float

            @model_validator(mode='after')
            def verify_square(self) -> Self:
                if self.width != self.height:
                    raise ValueError('width and height do not match')
                return self

        s = Square(width=1, height=1)
        print(repr(s))
        #> Square(width=1.0, height=1.0)

        try:
            Square(width=1, height=2)
        except ValidationError as e:
            print(e)
            '''
            1 validation error for Square
              Value error, width and height do not match [type=value_error, input_value={'width': 1, 'height': 2}, input_type=dict]
            '''
        ```

        For more in depth examples, see [Model Validators](../concepts/validators.md#model-validators).

        Parameters
        ----------
        mode: Literal["before", "after"]
            A required string literal that specifies the validation mode.
            It can be one of the following: 'before', or 'after'.

        Returns
        -------
        A decorator that can be used to decorate a function to be used as a model validator.
        """
        pre = mode == "before"
        return root_validator(pre=pre)

    def validate_call(
        func: Optional[_AnyCallable] = None,
        /,
        *,
        config: Optional[Dict] = None,
        validate_return: bool = False,
    ) -> Union[_AnyCallable, Callable[[_AnyCallable], _AnyCallable]]:
        """
        Returns a decorated wrapper around the function that validates the arguments and, optionally, the return value.
        Usage may be either as a plain decorator `@validate_call` or with arguments `@validate_call(...)`.

        Parameters
        ----------
        func : Optional[_AnyCallable], defaults to None
            The function to be decorated.
        config: Optional[Dict], defaults to None
            The configuration dictionary.
        validate_return: bool
            Whether to validate the return value.
            Placeholder for pydantic v2 compatibility, as this functionality is absent in v1.

        Returns
        -------
        Union[_AnyCallable, Callable[[_AnyCallable], _AnyCallable]]
            The decorated function.
        """
        if not hasattr(validate_call, "warning_raised") and validate_return:
            warn("validate_return is not supported in pydantic v1", UserWarning)
            validate_call.warning_raised = True

        for v1_name, v2_name in [
            ("allow_population_by_field_name", "populate_by_name"),
            ("anystr_lower", "str_to_lower"),
            ("anystr_strip_whitespace", "str_strip_whitespace"),
            ("anystr_upper", "str_to_upper"),
            ("keep_untouched", "ignored_types"),
            ("max_anystr_length", "str_max_length"),
            ("min_anystr_length", "str_min_length"),
            ("orm_mode", "from_attributes"),
            ("schema_extra", "json_schema_extra"),
            ("validate_all", "validate_default"),
        ]:
            config = _convert_config_param(config, v2_name, v1_name)

        return validate_arguments(func=func, config=config)

    def _convert_config_param(config: Dict[str, Any], v2_name: str, v1_name: str) -> Dict[str, Any]:
        """
        Convert a config parameter from v2 to v1.

        Parameters
        ----------
        config : Dict[str, Any]
            The dictionary of configuration parameters.
        v2_name : str
            The v2 name of the configuration parameter.
        v1_name : str
            The v1 name of the configuration parameter.

        Returns
        -------
        config : Dict[str, Any]
            The converted dictionary of configuration parameters.
        """
        if config is not None and v2_name in config:
            config[v1_name] = config.pop(v2_name)
        return config

elif pydantic_version == PYDANTIC_VERSION_2:
    from pydantic import (
        field_validator,
        model_validator,
        validate_call,
    )
else:
    raise ImportError(f"Unsupported pydantic version: {pydantic_version}")

__all__ = [
    "field_validator",
    "model_validator",
    "validate_call",
    "NonNegativeFloat",
    "NonNegativeInt",
    "PositiveInt",
    "BaseModel",
    "ValidationError",
    "confloat",
    "conint",
    "constr",
    "Field",
    "PrivateAttr",
]
