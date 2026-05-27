"""JSON serialization for IR dataclasses — enums as strings, nested structures."""

from __future__ import annotations

import json
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints, overload

T = TypeVar("T")


def _unwrap_optional(target_type: Any) -> tuple[Any, bool]:
    origin = get_origin(target_type)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(target_type) if arg is not type(None)]
        if len(args) == 1:
            return args[0], True
    return target_type, False


def _resolve_type(target_type: Any) -> Any:
    """Pick concrete type for unions like str | dict."""
    origin = get_origin(target_type)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(target_type) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
        # Prefer dataclass over primitives when deserializing objects
        for arg in args:
            if is_dataclass(arg):
                return arg
        return args[0]
    return target_type


def to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: to_dict(getattr(obj, field.name)) for field in fields(obj)}
    if isinstance(obj, dict):
        return {str(key): to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    return obj


def from_dict(data: Any, target_type: type[T]) -> T:
    if data is None:
        return None  # type: ignore[return-value]

    inner_type, _is_optional = _unwrap_optional(target_type)
    if _is_optional and data is None:
        return None  # type: ignore[return-value]

    target_type = inner_type
    origin = get_origin(target_type)

    if origin in (Union, types.UnionType):
        return from_dict(data, _resolve_type(target_type))  # type: ignore[return-value]

    if origin is list:
        (item_type,) = get_args(target_type)
        return [from_dict(item, item_type) for item in data]  # type: ignore[return-value]

    if origin is dict:
        key_type, value_type = get_args(target_type)
        return {
            from_dict(key, key_type): from_dict(value, value_type)
            for key, value in data.items()
        }  # type: ignore[return-value]

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(data)  # type: ignore[return-value]

    if is_dataclass(target_type):
        hints = get_type_hints(target_type)
        kwargs: dict[str, Any] = {}
        for field in fields(target_type):
            if field.name not in data:
                continue
            field_type = hints.get(field.name, field.type)
            inner, optional = _unwrap_optional(field_type)
            if optional and data[field.name] is None:
                kwargs[field.name] = None
            else:
                kwargs[field.name] = from_dict(data[field.name], inner)
        return target_type(**kwargs)  # type: ignore[return-value]

    return data  # type: ignore[return-value]


def to_json(obj: Any, *, indent: int | None = 2) -> str:
    return json.dumps(to_dict(obj), indent=indent, ensure_ascii=False)


@overload
def from_json(text: str, target_type: type[T]) -> T: ...


@overload
def from_json(text: str, target_type: None = None) -> Any: ...


def from_json(text: str, target_type: type[T] | None = None) -> T | Any:
    data = json.loads(text)
    if target_type is None:
        return data
    return from_dict(data, target_type)
