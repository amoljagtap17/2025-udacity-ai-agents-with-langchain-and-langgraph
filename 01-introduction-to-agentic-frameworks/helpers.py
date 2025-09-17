from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from typing import (
    List, Dict, Literal, 
    Callable, Any, 
    get_type_hints
)
import datetime
import inspect
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

model = "gpt-5-nano-2025-08-07"

temperature = 1.0

class Memory:
    def __init__(self):
        self._messages: List[Dict[str, str]] = []
    
    def add_message(self, 
                    role: Literal['user', 'system', 'assistant', 'tool'], 
                    content: str,
                    tool_calls: dict=dict(),
                    tool_call_id=None)-> None:

        message = {
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
        }

        if role == "tool":
            message = {
                "role": role,
                "content": content,
                "tool_call_id": tool_call_id,
            }

        self._messages.append(message)

    def get_messages(self) -> List[Dict[str, str]]:
        return self._messages

    def last_message(self) -> None:
        if self._messages:
            return self._messages[-1]

    def reset(self) -> None:
        self._messages = []

def chat_with_tools(user_question:str=None, 
                    memory:Memory=None, 
                    model:str=model, 
                    temperature=1.0, 
                    tools=None)-> str:
    
    messages = [{"role": "user", "content": user_question}]
    
    if memory:
        if user_question:
            memory.add_message(role="user", content=user_question)
        messages = memory.get_messages()        
    
    response = client.chat.completions.create(
        model = model,
        temperature = temperature,
        messages = messages,
        tools=tools,
    )
    
    ai_message = str(response.choices[0].message.content)
    tool_calls = response.choices[0].message.tool_calls
    
    if memory:
        memory.add_message(role="assistant", content=ai_message, tool_calls=tool_calls)
    
    return ai_message

def multiply(num1: float, num2: float) -> float:
    """Provides the result for multiplying two numbers"""

    return num1 * num2

class Tool:
    def __init__(self, func:Callable):
        self.func = func
        self.name = func.__name__
        self.description = func.__doc__
        self.argument_types_map = get_type_hints(func)
        self.signature = inspect.signature(func)
        self.arguments = [
            {
                "name": key, 
                "type": self._infer_json_schema_type(value),
                "required": param.default == inspect.Parameter.empty
            } 
            for key, value in self.argument_types_map.items()
            if (param := self.signature.parameters.get(key))
        ]

    def dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parallel_tool_calls": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        argument["name"]: {
                            "type": argument["type"],
                        }
                        for argument in self.arguments
                    },
                    "required": [
                        argument["name"] 
                        for argument in self.arguments 
                        if argument["required"]
                    ],
                    "additionalProperties": False,
                },
                "strict": True
            }
        }

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def _infer_json_schema_type(self, arg_type: Any) -> str:
        if arg_type == bool:
            return "boolean"
        elif arg_type == int:
            return "integer"
        elif arg_type == float:
            return "number"
        elif arg_type == str:
            return "string"
        elif arg_type == list:
            return "array"
        elif arg_type == dict:
            return "object"
        elif arg_type is None:
            return "null"
        elif arg_type == datetime.date or arg_type == datetime.datetime:
            return "string"  # JSON Schema treats dates as strings
        else:
            return "string"  # Default to string if type is unknown
        
class Agent:
    """A tool-calling AI Agent"""

    def __init__(
        self,
        name:str = "Agent", 
        role:str = "Personal Assistant",
        instructions:str = "Assist the user with their queries",
        model:str = model,
        temperature:float = temperature,
        tools:List[Tool] = [],
    ):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.model = model
        self.temperature = temperature
        self.memory = Memory()
        self.memory.add_message(
            role="system",
            content=f"You're an helpful AI Agent, your role is {self.role}, " 
                    f"and you need to {self.instructions}",
        )

        self.client = OpenAI()

        self.tools = tools
        self.tool_map = {t.name:t for t in tools}
        self.openai_tools = [t.dict() for t in self.tools] if self.tools else None

    def invoke(self, user_message: str) -> str:
        self.memory.add_message(
            role="user",
            content=user_message,
        )

        ai_message = self._get_completion(
            messages = self.memory.get_messages(),
        )

        tool_calls = ai_message.tool_calls
        
        self.memory.add_message(
            role="assistant",
            content=ai_message.content,
            tool_calls=tool_calls,
        )

        if tool_calls:
            self._call_tools(tool_calls)
            
        return self.memory.last_message()

    def _call_tools(self, tool_calls:List[ChatCompletionMessageToolCall]):
        for t in tool_calls:
            tool_call_id = t.id
            function_name = t.function.name
            args = json.loads(t.function.arguments)
            callable_tool = self.tool_map[function_name]
            result = callable_tool(**args)
            self.memory.add_message(
                role="tool", 
                content=str(result), 
                tool_call_id=tool_call_id
            )

        ai_message = self._get_completion(
            messages = self.memory.get_messages(),
        )

        tool_calls = ai_message.tool_calls

        self.memory.add_message(
            role="assistant",
            content=ai_message.content,
            tool_calls=tool_calls,
        )

        if tool_calls:
            self._call_tools(tool_calls)


    def _get_completion(self, messages:List[Dict])-> ChatCompletionMessage:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=self.openai_tools,
        )
        
        return response.choices[0].message
