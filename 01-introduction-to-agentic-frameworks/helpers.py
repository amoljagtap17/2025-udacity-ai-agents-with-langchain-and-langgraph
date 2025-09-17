from typing import List, Dict, Literal
from openai import OpenAI
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
    """Multiplies two numbers and returns the result."""

    return num1 * num2