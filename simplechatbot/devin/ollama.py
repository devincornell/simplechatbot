from __future__ import annotations

import typing
import dataclasses


# BaseChatModel
from langchain_ollama import ChatOllama

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool, BaseToolkit

from .chatbot import ChatBot


class OpenAIChatBot(ChatBot):
    '''Chatbot created from an Ollama model. Only separate so that it can be imported separately for dependency reasons.'''
    @classmethod
    def from_ollama(cls,
        model_name: str = "llama3.1", 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_factory: typing.Optional[typing.Callable[[BaseChatModel],list[BaseTool]]] = None,
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
        model = ChatOllama(
            model=model_name, 
            **model_kwargs
        )
        return cls.from_model(
            model = model,
            system_prompt = system_prompt,
            tools = tools,
            toolkits = toolkits,
            tool_factory = tool_factory,
        )
    
