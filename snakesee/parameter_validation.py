"""Parameter validation decorators and utilities.

This module provides validation decorators and functions for checking
function arguments and parameters.

Example:
    @validate_positive("half_life_days", "half_life_logs")
    def weighted_mean(self, half_life_days=7.0, half_life_logs=10):
        ...
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import ParamSpec
from typing import TypeVar

from snakesee.exceptions import InvalidParameterError

P = ParamSpec("P")
R = TypeVar("R")


# =============================================================================
# Validation Decorators
# =============================================================================


def validate_positive(*param_names: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that specified parameters are positive.

    Args:
        *param_names: Names of parameters that must be positive (> 0).

    Returns:
        Decorator that validates the specified parameters.

    Raises:
        InvalidParameterError: If any specified parameter is not positive.

    Note:
        None values are allowed and bypass validation, making this suitable
        for optional parameters with None defaults.

    Example:
        @validate_positive("half_life_days", "half_life_logs")
        def weighted_mean(self, half_life_days=7.0, half_life_logs=10):
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Cache signature at decoration time for better performance
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            for name in param_names:
                if name in bound.arguments:
                    value = bound.arguments[name]
                    if value is not None and value <= 0:
                        raise InvalidParameterError(
                            parameter=name,
                            value=value,
                            constraint="positive (> 0)",
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_non_negative(*param_names: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that specified parameters are non-negative.

    Args:
        *param_names: Names of parameters that must be non-negative (>= 0).

    Returns:
        Decorator that validates the specified parameters.

    Raises:
        InvalidParameterError: If any specified parameter is negative.

    Note:
        None values are allowed and bypass validation, making this suitable
        for optional parameters with None defaults.

    Example:
        @validate_non_negative("timeout")
        def wait_for_job(self, timeout=30.0):
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Cache signature at decoration time for better performance
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            for name in param_names:
                if name in bound.arguments:
                    value = bound.arguments[name]
                    if value is not None and value < 0:
                        raise InvalidParameterError(
                            parameter=name,
                            value=value,
                            constraint="non-negative (>= 0)",
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_not_empty(*param_names: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that specified sequence parameters are not empty.

    Args:
        *param_names: Names of parameters that must not be empty.

    Returns:
        Decorator that validates the specified parameters.

    Raises:
        InvalidParameterError: If any specified parameter is empty.

    Note:
        None values are allowed and bypass validation, making this suitable
        for optional parameters with None defaults.

    Example:
        @validate_not_empty("durations")
        def calculate_stats(self, durations):
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Cache signature at decoration time for better performance
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            for name in param_names:
                if name in bound.arguments:
                    value = bound.arguments[name]
                    if value is not None and len(value) == 0:
                        raise InvalidParameterError(
                            parameter=name,
                            value=value,
                            constraint="non-empty",
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_in_range(
    param_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to validate that a parameter is within a specified range.

    Args:
        param_name: Name of the parameter to validate.
        min_value: Minimum allowed value (inclusive). None means no minimum.
        max_value: Maximum allowed value (inclusive). None means no maximum.

    Returns:
        Decorator that validates the specified parameter.

    Raises:
        InvalidParameterError: If the parameter is outside the range.

    Note:
        None values are allowed and bypass validation, making this suitable
        for optional parameters with None defaults.

    Example:
        @validate_in_range("confidence", min_value=0.0, max_value=1.0)
        def set_confidence(self, confidence):
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Cache signature at decoration time for better performance
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            if param_name in bound.arguments:
                value = bound.arguments[param_name]
                if value is not None:
                    if min_value is not None and value < min_value:
                        raise InvalidParameterError(
                            parameter=param_name,
                            value=value,
                            constraint=f">= {min_value}",
                        )
                    if max_value is not None and value > max_value:
                        raise InvalidParameterError(
                            parameter=param_name,
                            value=value,
                            constraint=f"<= {max_value}",
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Inline Validation Functions
# =============================================================================


def require_positive(value: float | int, name: str) -> float | int:
    """Validate that a value is positive.

    Args:
        value: The value to check (must not be None).
        name: Parameter name for error messages.

    Returns:
        The value if valid.

    Raises:
        InvalidParameterError: If value is None or not positive.
    """
    if value is None:
        raise InvalidParameterError(parameter=name, value=value, constraint="not None")
    if value <= 0:
        raise InvalidParameterError(parameter=name, value=value, constraint="positive (> 0)")
    return value


def require_non_negative(value: float | int, name: str) -> float | int:
    """Validate that a value is non-negative.

    Args:
        value: The value to check (must not be None).
        name: Parameter name for error messages.

    Returns:
        The value if valid.

    Raises:
        InvalidParameterError: If value is None or negative.
    """
    if value is None:
        raise InvalidParameterError(parameter=name, value=value, constraint="not None")
    if value < 0:
        raise InvalidParameterError(parameter=name, value=value, constraint="non-negative (>= 0)")
    return value


def require_not_empty(value: Sequence[Any], name: str) -> Sequence[Any]:
    """Validate that a sequence is not empty.

    Args:
        value: The sequence to check (must not be None).
        name: Parameter name for error messages.

    Returns:
        The value if valid.

    Raises:
        InvalidParameterError: If value is None or sequence is empty.
    """
    if value is None:
        raise InvalidParameterError(parameter=name, value=value, constraint="not None")
    if len(value) == 0:
        raise InvalidParameterError(parameter=name, value=value, constraint="non-empty")
    return value


def require_in_range(
    value: float | int,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | int:
    """Validate that a value is within a range.

    Args:
        value: The value to check (must not be None).
        name: Parameter name for error messages.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).

    Returns:
        The value if valid.

    Raises:
        InvalidParameterError: If value is None or outside the range.
    """
    if value is None:
        raise InvalidParameterError(parameter=name, value=value, constraint="not None")
    if min_value is not None and value < min_value:
        raise InvalidParameterError(parameter=name, value=value, constraint=f">= {min_value}")
    if max_value is not None and value > max_value:
        raise InvalidParameterError(parameter=name, value=value, constraint=f"<= {max_value}")
    return value
