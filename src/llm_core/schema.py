# -*- coding: utf-8 -*-
import re
import json
import dataclasses
import typing
from enum import Enum

from llama_cpp.llama_grammar import LlamaGrammar


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
        elif field_type.__name__ == "set":
            return {
                "type": "array",
                "uniqueItems": True,
                "items": get_type(field_type.__args__[0]),
            }
        elif field_type.__name__ == "list":
            return {
                "type": "array",
                "items": get_type(field_type.__args__[0]),
            }
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
        elif issubclass(field_type, Enum):
            return {
                "type": "string",
                "enum": list(field_type.__members__.keys()),
            }
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


def from_dict(cls, data):
    if dataclasses.is_dataclass(cls):
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        return cls(
            **{k: from_dict(field_types[k], v) for k, v in data.items()}
        )
    elif cls.__name__ == "list":
        return [from_dict(cls.__args__[0], v) for v in data]
    elif cls.__name__ == "set":
        return set([from_dict(cls.__args__[0], v) for v in data])
    elif isinstance(cls, typing._GenericAlias):
        if cls._name == "List":
            return [from_dict(cls.__args__[0], v) for v in data]
        elif cls._name == "Dict":
            return {k: from_dict(cls.__args__[1], v) for k, v in data.items()}
    elif issubclass(cls, Enum):
        return getattr(cls, data)
    else:
        return data


SPACE_RULE = '" "?'

PRIMITIVE_RULES = {
    "boolean": '("true" | "false") space',
    "number": '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    "integer": '("-"? ([0-9] | [1-9] [0-9]*)) space',
    "string": r""" "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space """,
    "null": '"null" space',
}

INVALID_RULE_CHARS_RE = re.compile(r"[^a-zA-Z0-9-]+")
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_LITERAL_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"'}


class SchemaConverter:
    def __init__(self, prop_order):
        self._prop_order = prop_order
        self._rules = {"space": SPACE_RULE}

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)),
            json.dumps(literal),
        )
        return f'"{escaped}"'

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub("-", name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f"{esc_name}{i}" in self._rules:
                i += 1
            key = f"{esc_name}{i}"
        self._rules[key] = rule
        return key

    def visit(self, schema, name):
        schema_type = schema.get("type")
        rule_name = name or "root"

        if "oneOf" in schema or "anyOf" in schema:
            rule = " | ".join(
                (
                    self.visit(alt_schema, f'{name}{"-" if name else ""}{i}')
                    for i, alt_schema in enumerate(
                        schema.get("oneOf") or schema["anyOf"]
                    )
                )
            )
            return self._add_rule(rule_name, rule)

        elif "const" in schema:
            return self._add_rule(
                rule_name, self._format_literal(schema["const"])
            )

        elif "enum" in schema:
            rule = " | ".join(
                (self._format_literal(v) for v in schema["enum"])
            )
            return self._add_rule(rule_name, rule)

        elif schema_type == "object" and "properties" in schema:
            # TODO: `required` keyword
            prop_order = self._prop_order
            prop_pairs = sorted(
                schema["properties"].items(),
                # sort by position in prop_order (if specified) then by key
                key=lambda kv: (prop_order.get(kv[0], len(prop_order)), kv[0]),
            )

            rule = '"{" space'
            for i, (prop_name, prop_schema) in enumerate(prop_pairs):
                prop_rule_name = self.visit(
                    prop_schema, f'{name}{"-" if name else ""}{prop_name}'
                )
                if i > 0:
                    rule += ' "," space'
                rule += rf' {self._format_literal(prop_name)} space ":" space {prop_rule_name}'
            rule += ' "}" space'

            return self._add_rule(rule_name, rule)

        elif schema_type == "array" and "items" in schema:
            # TODO `prefixItems` keyword
            item_rule_name = self.visit(
                schema["items"], f'{name}{"-" if name else ""}item'
            )
            rule = f'"[" space ({item_rule_name} ("," space {item_rule_name})*)? "]" space'
            return self._add_rule(rule_name, rule)

        else:
            assert (
                schema_type in PRIMITIVE_RULES
            ), f"Unrecognized schema: {schema}"
            return self._add_rule(
                "root" if rule_name == "root" else schema_type,
                PRIMITIVE_RULES[schema_type],
            )

    def format_grammar(self):
        return "\n".join(
            (f"{name} ::= {rule}" for name, rule in self._rules.items())
        )


def to_grammar(schema):
    converter = SchemaConverter({})
    converter.visit(schema, "")
    grammar_string = converter.format_grammar()
    return LlamaGrammar.from_string(grammar_string, verbose=False)
