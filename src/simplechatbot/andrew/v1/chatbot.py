

from langchain_core.language_models import BaseChatModel
from langchain_community.utilities import SQLDatabase
import langchain_core.tools
from langchain_core.tools import BaseTool


from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)

import dataclasses
import typing
from typing import Optional


## Chatbot


## ChatBedrock specific
def add_cost(inp_msg: AIMessage) -> float:
    return inp_msg.additional_kwargs['usage'].get('prompt_tokens')*(3/1e6) + inp_msg.additional_kwargs['usage'].get('completion_tokens')*(15/1e6)
    
    
## LLMWithHistory 
@dataclasses.dataclass
class LLMWithHistory:
    llm: BaseChatModel
    history: list[BaseMessage] = dataclasses.field(default_factory=list)
    system_message: Optional[str] = None  # Optional system message
    tools: typing.Optional[list[BaseTool]] = None
    rag: bool = False
    cost: float = 0

    def from_list(self, tool_list: list[langchain_core.tools.BaseTool]):
        self.tool_dict = {t.name:t for t in tool_list}
    
    def __post_init__(self):
        # Bind tools if there are any
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
            self.from_list(self.tools)
          
    def send_message_to_llm(self, message: str):
        # Put system prompt at beginning if one has been provided 
        if not len(self.history):
            if self.system_message:
                self.history.append(SystemMessage(self.system_message))
        
        # Trim messages so that the number of human messages in history never exceeds three
        # (history will never be more than: system message + (human + other stuff) + (human + other stuff) + (human + whatever is in latest response)
        human_msg_lst = [type(msg) for msg in self.history if type(msg) == langchain_core.messages.human.HumanMessage]
        if len(human_msg_lst) > 2:
            human_msg_idx = [idx for idx, msg in enumerate(self.history) if type(msg) == langchain_core.messages.human.HumanMessage]
            self.history = self.history[:1] + self.history[human_msg_idx[-2]:]
            
        human_message = HumanMessage(message)
        
        # Append the human message to the history
        self.history.append(human_message)
        
        # Invoke the LLM with the current history
        response = self.llm.invoke(self.history)
        self.history.append(response)
        # self.cost += add_cost(response)
        
        # Call tools if necessary 
        while response.response_metadata['finish_reason'] == "tool_calls":
            
            for tool_info in response.tool_calls:
                
                # Get tool name 
                tool_name = tool_info['name']
                print(f"Running {tool_name} tool...\n")
                
                tool_args = tool_info['args']
                tool_output = self.tool_dict[tool_name].invoke(tool_args)
                
                tool_response = ToolMessage(content=tool_output, 
                            tool_call_id=tool_info['id'])
                
                self.history.append(tool_response)
                
            response = self.llm.invoke(self.history)
            # self.cost += add_cost(response)

            # Append the response to the history
            self.history.append(response)
                
        # "Stream" the response 
        for fake_token in response.content:
            print(fake_token, end="", flush=True)

        
