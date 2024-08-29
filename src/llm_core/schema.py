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

        elif isinstance(field_type, GenericAlias):
            return convert_generic_alias(field_type)

        elif (
            hasattr(field_type, "__origin__")
            and field_type.__origin__ is Union
        ):
            return convert_union(field_type)

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
                    f"{f.name} ({f.type.__name__}) "
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
        step_1_query_analysis: str
        step_1_function_name: ProviderName
        step_1_function_arguments: reduce(lambda a, b: {**a, **b}, providers)

        identified_entities: List[str]
        missing_entities: List[str]
        identified_implications: List[str]
        identified_hypothesis: List[str]

        step_2_analysis_evaluation: str
        step_2_revised_plan: str
        step_2_function_name: ProviderName
        step_2_function_arguments: reduce(lambda a, b: {**a, **b}, providers)

        def format_results(self, results):
            return dedent(
                f"""# Detailed Plan

                ## Step 1: Query Analysis
                Query Analysis: {self.step_1_query_analysis}

                Function Name: {self.step_1_function_name}

                Function Arguments: {self.step_1_function_arguments}

                ## Identified Entities
                {', '.join(self.identified_entities)}

                ## Missing Entities
                {', '.join(self.missing_entities)}

                ## Identified Implications
                {', '.join(self.identified_implications)}

                ## Identified Hypothesis
                {', '.join(self.identified_hypothesis)}

                ## Step 2: Analysis Evaluation
                Analysis Evaluation: {self.step_2_analysis_evaluation}

                Revised Plan: {self.step_2_revised_plan}

                Function Name: {self.step_2_function_name}

                Function Arguments: {self.step_2_function_arguments}

                ## Execution trace

                Results:

                {results}
                """
            )

    @dataclass
    class SelectionTool:
        prompt = """
        Design a very detailed plan to fulfill the user's query.

        # Step 1

        - Analyze user's query and select a function
        - Write the function arguments

        # Step 2

        - Evaluate the analysis of "Step 1"
        - Write a revised plan with more details

        # Guidelines

        Entities are:
        - specific
        - present or missing

        Relations are:
        - implications
        - hypothesis
        """
        detailed_plan: DetailedPlan

        def execute(self):
            trace = []

            function_name = self.detailed_plan.step_2_function_name.name

            trace.append(f"The execution of the function: {function_name}")

            #: This first branch is run when only one tool is available
            #: In that case, step_2_function_arguments contains a populated
            #: dataclass
            if is_dataclass(self.detailed_plan.step_2_function_arguments):
                arguments = asdict(
                    self.detailed_plan.step_2_function_arguments
                )
                executable_partial = (
                    self.detailed_plan.step_2_function_arguments
                )

            #: This second branch is run when several tools are available
            #: In that case, step_2_function_arguments contains a mapping
            else:
                arguments = self.detailed_plan.step_2_function_arguments
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
