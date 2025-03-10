from __future__ import annotations

import typing
import dataclasses


# BaseChatModel
from langchain_mistralai import ChatMistralAI

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool, BaseToolkit
    from ..agent.toolset import ToolFactoryType, ToolName

from ..agent import Agent


class MistralAgent(Agent):
    '''Chatbot created from an Mistral model. Only separate so that it can be imported separately for dependency reasons.'''
    @classmethod
    def new(cls,
        model_name: str = "mistral-large-latest", 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_factories: list[ToolFactoryType] | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None,
        **model_kwargs,
    ) -> typing.Self:
        '''Create a new chatbot with an ollama model.
        Args:
            model_name: name of the model to use
            system_prompt: first system message for the chat.
            tools: tools to be bound to the model using model.bind_tools(tools).
            tool_callable: function to get tools to use. Here so that the tools can access a reference to the model.
            model_kwargs: any additional arguments to pass to the model constructor.
        '''
        model = ChatMistralAI(
            model=model_name, 
            **model_kwargs
        )
        return cls.from_model(
            model = model,
            system_prompt = system_prompt,
            tools = tools,
            toolkits = toolkits,
            tool_factories=tool_factories,
            tool_choice = tool_choice,
        )
    
