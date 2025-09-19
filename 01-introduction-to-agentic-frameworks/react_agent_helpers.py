import datetime
import inspect
import json
from typing import (
    List,
    Dict,
    Literal,
    Callable,
    Any,
    get_type_hints
)
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

load_dotenv()

model = "gpt-5-nano-2025-08-07"

temperature = 1.0

# Memory class to store conversation history 
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


# Tool wrapper to convert a function into a tool with metadata
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

# Exception to stop the ReAct loop 
class StopReactLoopException(Exception):
    """
    Terminates ReAct loop
    """

TERMINATION_MESSAGE = "StopReactLoopException"

# Function to signal termination of the ReAct loop
def termination()-> str:
    """Terminate the ReAct loop. If the agent thinks there's no further actions to take"""
    return TERMINATION_MESSAGE

# Main Agent class implementing the ReAct framework
class Agent:
    """A ReAct AI Agent"""
    
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
        self.client = OpenAI()
        self.tools = tools
        self.termination_message = TERMINATION_MESSAGE
        self._register_tool(Tool(termination))
        self.tool_map = {t.name:t for t in tools}
        self.openai_tools = [t.dict() for t in self.tools] if self.tools else None
        self.memory = Memory()

        self.memory.add_message(
            role="system",
            content=f"You're an AI Agent, your role is {self.role}, " 
                    f"and you need to {self.instructions} "
                    "You can answer multistep questions by sequentially calling functions. "
                    "You follow a pattern of of Thought and Action. "
                    "Create a plan of execution: "
                    "- Use Thought to describe your thoughts about the question you have been asked. "
                    "- Use Action to specify one of the tools available to you. "
                    "When you think it's over call the termination tool. "
                    "Never try to respond directly if the question needs a tool."
                    "The actions you have are the Tools: "
                    f"{self.openai_tools}",
        )

    def invoke(self, user_message: str, max_iter:int=3) -> str:
        self.memory.add_message(
            role="user",
            content=user_message,
        )
        try:
            self._react_loop(max_iter)
        except StopReactLoopException as e:
            print(f"Termninated loop with message: '{e}'")
            self._reason()

        return self.memory.last_message()

    def _react_loop(self, max_iter:int):
        for i in range(max_iter):
            self._reason()

            ai_message = self._get_completion(
                messages = self.memory.get_messages(),
                tools=self.openai_tools,
            )
            tool_calls = ai_message.tool_calls

            self.memory.add_message(
                role="assistant",
                content=ai_message.content,
                tool_calls=tool_calls,
            )

            if tool_calls:
                self._call_tools(tool_calls)

    def _reason(self):
        # No tools
        ai_message = self._get_completion(
            messages = self.memory.get_messages(),
        )

        self.memory.add_message(
            role="assistant",
            content=ai_message.content,
            tool_calls=None,
        )

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
            if result == TERMINATION_MESSAGE:
                raise StopReactLoopException(TERMINATION_MESSAGE)

    def _get_completion(self, messages:List[Dict], tools:List=None)-> ChatCompletionMessage:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=tools,
        )
        
        return response.choices[0].message

    def _register_tool(self, tool:Tool):
        self.tools.append(tool)


# Example tool functions
def add(num1: int, num2: int) -> int:
    """Adds two numbers"""
    return num1 + num2

def multiply(num1: int, num2: int) -> int:
    """Multiplies two numbers"""
    return num1 * num2

def subtract(num1: int, num2: int) -> int:
    """Subtracts two numbers"""
    return num1 - num2