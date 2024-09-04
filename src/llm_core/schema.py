# -*- coding: utf-8 -*-
import json
from typing import Union, List, Dict, Set, Tuple, FrozenSet
from typing import _GenericAlias as GenericAlias

from enum import Enum
from functools import reduce
from dataclasses import (
    dataclass,
    fields,
    asdict,
    is_dataclass,
    MISSING,
)
from textwrap import dedent, indent
from llama_cpp.llama_grammar import LlamaGrammar


def convert_native_container_type(container_type, items_type):
    items = convert_field_type(items_type)

    mapping = {
        List: {"type": "array", "items": items},
        Tuple: {"type": "array", "items": items},
        Dict: {"type": "object", "additionalProperties": items},
        Set: {"type": "array", "uniqueItems": True, "items": items},
        FrozenSet: {"type": "array", "uniqueItems": True, "items": items},
        list: {"type": "array", "items": items},
        tuple: {"type": "array", "items": items},
        dict: {"type": "object", "additionalProperties": items},
        set: {"type": "array", "uniqueItems": True, "items": items},
        frozenset: {"type": "array", "uniqueItems": True, "items": items},
    }
    return mapping[container_type]


def convert_generic_alias(field_type):
    container_type = field_type.__origin__
    items_type = field_type.__args__

    if len(items_type) != 1:
        print(field_type)
        print(items_type)
        raise NotImplementedError("Complex annotations are not supported")

    items_type = items_type[0]

    return convert_native_container_type(container_type, items_type)


def convert_union(field_type):
    available_types = field_type.__args__
    return {"anyOf": [convert_field_type(t) for t in available_types]}


def convert_complex_field_type(field_type):
    try:
        if is_dataclass(field_type):
            return to_json_schema(field_type)

        elif (
            hasattr(field_type, "__origin__")
            and field_type.__origin__ is Union
        ):
            return convert_union(field_type)

        elif isinstance(field_type, GenericAlias):
            return convert_generic_alias(field_type)

        elif issubclass(field_type, Enum):
            return {
                "type": "string",
                "enum": list(field_type.__members__.keys()),
            }

        else:  #: Let the possibility of having a non specified object
            return {
                "type": "object",
            }
    except Exception as e:
        print("Error", field_type, is_dataclass(field_type), type(field_type))
        raise e


def convert_field_type(field_type):
    mapping = {
        int: {"type": "integer"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        float: {"type": "number"},
        complex: {"type": "string", "format": "complex-number"},
        bytes: {"type": "string", "contentEncoding": "base64"},
    }

    if field_type in mapping:
        return mapping[field_type]
    else:
        return convert_complex_field_type(field_type)


def to_json_schema(datacls):
    if callable(getattr(datacls, "json_schema", None)):
        return datacls.json_schema()

    properties = {}
    required_fields = []
    for field in fields(datacls):
        properties[field.name] = convert_field_type(field.type)

        no_default = field.default == MISSING
        no_default_factory = field.default_factory == MISSING
        required = no_default and no_default_factory

        if required:
            required_fields.append(field.name)

    return {
        "type": "object",
        "title": datacls.__name__,
        "description": datacls.__doc__,
        "properties": properties,
        "required": required_fields,
    }


def from_dict(cls, attrs):
    sequences = (
        list,
        tuple,
        set,
        frozenset,
    )
    try:
        if is_dataclass(cls):
            field_types = {f.name: f.type for f in fields(cls)}
            return cls(
                **{k: from_dict(field_types[k], v) for k, v in attrs.items()}
            )

        elif hasattr(cls, "__origin__") and cls.__origin__ is Union:
            return attrs

        elif hasattr(cls, "__origin__") and cls.__origin__ in sequences:
            return cls.__origin__(
                [from_dict(cls.__args__[0], v) for v in attrs]
            )

        elif hasattr(cls, "__name__") and cls.__name__ == "list":
            return [from_dict(cls.__args__[0], v) for v in attrs]

        elif hasattr(cls, "__name__") and cls.__name__ == "set":
            return set([from_dict(cls.__args__[0], v) for v in attrs])

        elif issubclass(cls, Enum):
            return getattr(cls, attrs)

        else:
            return attrs
    except AttributeError as e:
        print(f"Error on {cls}")
        raise e


def to_grammar(schema):
    return LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)


def make_tool(datacls):
    interface_schema = to_json_schema(datacls)
    tool_schema = {
        "type": "object",
        "properties": {
            "name": datacls.__name__,
            "description": datacls.__doc__.strip(),
            "parameters": interface_schema,
        },
    }
    return tool_schema


def make_tools(providers):
    return [to_json_schema(provider) for provider in providers]


def make_helper(provider):
    name = f"name: {provider.__name__}"
    doc = f"description: {dedent(provider.__doc__.strip())}"
    light_schema = indent(
        "\n".join(
            [
                (
                    f"{f.name} ({getattr(f.type, '__name__', getattr(f.type, '_name', ''))}) "
                    f"{'(required)' if f.default == MISSING and f.default_factory == MISSING else ''}"
                )
                for f in fields(provider)
            ]
        ),
        "  ",
    )

    return "\n".join((name, doc, "inputs:", light_schema))


def make_tool_helper(providers):
    return "\n------\n".join(
        ["Available tools:"]
        + [make_helper(provider) for provider in providers]
    )


def make_selection_tool(providers):
    providers_registry = {
        provider.__name__: provider for provider in providers
    }
    ProviderName = Enum(
        "FunctionName", [provider.__name__ for provider in providers]
    )

    @dataclass
    class DetailedPlan:
        query_analysis: str
        entities: List[str]
        missing_entities: List[str]
        relations: List[str]
        missing_relations: List[str]
        plan: str

        function_name: ProviderName
        function_arguments: reduce(lambda a, b: Union[a, b], providers)

        def render(self, results):
            return dedent(
                f"""
                # Detailed Plan

                1. Analyze user's query

                {self.query_analysis}

                2. Identify entities

                {', '.join(self.entities)}

                3. Identify missing entities

                {', '.join(self.missing_entities)}

                4. Identify relations

                {', '.join(self.relations)}

                5. Identify missing relations

                {', '.join(self.missing_relations)}

                6. Write a concise plan

                {self.plan}

                7. Write the function name to use

                {self.function_name}

                8. Write the function arguments

                -- hidden for brevity purpose --

                ## Execution trace

                Results:

                {results}
                """
            ).strip()

    @dataclass
    class SelectionTool:
        prompt = """
        Design a plan to fulfill the user's query.

        Here's a framework to follow:

        1. Analyze user's query
        2. Identify entities
        3. Identify missing entities
        4. Identify relations
        5. Identify missing relations
        6. Write a concise plan
        7. Write the function name to use
        8. Write the function arguments

        """
        detailed_plan: DetailedPlan

        def execute(self):
            trace = []

            function_name = self.detailed_plan.function_name.name

            trace.append(f"The execution of the function: {function_name}")

            #: This first branch is run when only one tool is available
            #: In that case, function_arguments contains a populated
            #: dataclass
            if is_dataclass(self.detailed_plan.function_arguments):
                arguments = asdict(self.detailed_plan.function_arguments)
                executable_partial = self.detailed_plan.function_arguments

            #: This second branch is run when several tools are available
            #: In that case, function_arguments contains a mapping
            else:
                arguments = self.detailed_plan.function_arguments
                executable_partial = from_dict(
                    providers_registry[function_name], arguments
                )

            formatted_arguments = ",".join(
                [f"{k}={v}" for k, v in arguments.items()]
            )

            trace.append(f"with arguments: {formatted_arguments}")

            result = executable_partial()

            trace.append(f"gave the result: `{result}`")
            return " ".join(trace)

    schema = to_json_schema(SelectionTool)
    grammar = to_grammar(schema)

    SelectionTool.schema = schema
    SelectionTool.grammar = grammar
    SelectionTool.helpers = make_tool_helper(providers)

    return SelectionTool


def as_tool(json_schema):
    return {
        "type": "function",
        "function": {
            "name": json_schema["title"],
            "description": json_schema["description"],
            "parameters": json_schema,
        },
    }
