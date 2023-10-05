# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Choice:
    text: Optional[str] = None
    index: int
    logprobs: Optional[str] = None
    finish_reason: str


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class TextCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatCompletionChoice:
    index: int
    message: Message
    finish_reason: str


@dataclass
class ChatCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


{
    "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "object": "text_completion",
    "created": 1679561337,
    "model": "./models/7B/llama-model.gguf",
    "choices": [
        {
            "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 28,
        "total_tokens": 42,
    },
}

{
    "id": "chatcmpl-86EBKy1yobbD4bDk1wn5eggsRtjeU",
    "object": "chat.completion",
    "created": 1696495174,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Sure! Here's a classic pizza recipe for you:\n\nIngredients:\n- 1 pizza dough (store-bought or homemade)\n- 1 cup pizza sauce\n- 2 cups shredded mozzarella cheese\n- Toppings of your choice (e.g. pepperoni, mushrooms, bell peppers, onions, olives, etc.)\n- Olive oil, for brushing\n- Salt and pepper, to taste\n- Italian seasoning, for sprinkling (optional)\n\nInstructions:\n1. Preheat your oven to 475\u00b0F (245\u00b0C). If you have a pizza stone, place it in the oven while preheating.\n\n2. Roll out the pizza dough on a lightly floured surface to your desired thickness. If using store-bought dough, follow the instructions on the package.\n\n3. Transfer the rolled-out dough onto a pizza peel or a baking sheet lined with parchment paper.\n\n4. Brush the surface of the dough with olive oil, then spread the pizza sauce evenly over the dough, leaving a small border around the edges.\n\n5. Sprinkle half of the shredded mozzarella cheese over the sauce.\n\n6. Add your desired toppings, distributing them evenly over the pizza.\n\n7. Sprinkle the remaining mozzarella cheese over the toppings.\n\n8. Season with salt, pepper, and Italian seasoning, if desired.\n\n9. Carefully transfer the pizza onto the preheated pizza stone in the oven, or place the baking sheet directly in the oven.\n\n10. Bake for about 12-15 minutes, or until the crust is golden brown and the cheese is melted and bubbly.\n\n11. Remove the pizza from the oven and let it cool for a few minutes before slicing and serving.\n\nEnjoy your homemade pizza!",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 345,
        "total_tokens": 365,
    },
}
