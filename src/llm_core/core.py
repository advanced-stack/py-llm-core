# -*- coding: utf-8 -*-
import dirtyjson

from .schema import to_json_schema, from_dict


class BaseParser:
    def __init__(self, target_cls, *args, **kwargs):
        self.target_cls = target_cls
        parameters = to_json_schema(self.target_cls)
        self.schema = {
            "name": "PublishAnswer",
            "description": "Publish the answer",
            "parameters": parameters,
        }

    def parse(self, text: str):
        raise NotImplementedError

    def deserialize(self, json_str: str):
        attrs = dirtyjson.loads(json_str)
        instance = from_dict(self.target_cls, attrs)
        return instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
