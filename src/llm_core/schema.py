# -*- coding: utf-8 -*-
import dataclasses
import typing


def to_json_schema(datacls):
    def get_type(field_type):
        if field_type == int:
            return {"type": "integer"}
        elif field_type == str:
            return {"type": "string"}
        elif field_type == bool:
            return {"type": "boolean"}
        elif field_type == float:
            return {"type": "number"}
        elif field_type == complex:
            return {"type": "string", "format": "complex-number"}
        elif field_type == bytes:
            return {"type": "string", "contentEncoding": "base64"}
        elif field_type == tuple:
            return {"type": "array", "items": {}}
        elif field_type == set:
            return {"type": "array", "uniqueItems": True, "items": {}}
        elif isinstance(field_type, typing._GenericAlias):
            if field_type._name == "List":
                return {
                    "type": "array",
                    "items": get_type(field_type.__args__[0]),
                }
            elif field_type._name == "Dict":
                return {
                    "type": "object",
                    "additionalProperties": get_type(field_type.__args__[1]),
                }
        elif dataclasses.is_dataclass(field_type):
            return to_json_schema(field_type)
        else:
            return {"type": "object"}

    properties = {}
    required = []
    for field in dataclasses.fields(datacls):
        properties[field.name] = get_type(field.type)
        if (
            field.default == dataclasses.MISSING
            and field.default_factory == dataclasses.MISSING
        ):
            required.append(field.name)

    return {"type": "object", "properties": properties, "required": required}


def from_dict(datacls, data):
    if dataclasses.is_dataclass(datacls):
        field_types = {f.name: f.type for f in dataclasses.fields(datacls)}
        return datacls(
            **{k: from_dict(field_types[k], v) for k, v in data.items()}
        )
    elif isinstance(datacls, typing._GenericAlias):
        if datacls._name == "List":
            return [from_dict(datacls.__args__[0], v) for v in data]
        elif datacls._name == "Dict":
            return {
                k: from_dict(datacls.__args__[1], v) for k, v in data.items()
            }
    else:
        return data
