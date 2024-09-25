from __future__ import annotations

import typing
import dataclasses

import langchain_core.tools

# BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .message_history import MessageHistory
from .toolset import ToolSet, ToolCallResult

from .ui import ChatBotUI

#if typing.TYPE_CHECKING:
from langchain_core.messages import AIMessageChunk, AIMessage

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool


@dataclasses.dataclass
class ChatBot:
    '''Methods for working with a chatbot with history.
    Created mostly from following this tutorial:
        https://python.langchain.com/v0.2/docs/tutorials/chatbot/

    Note that I'm only using a subset of the features there since I thought it'd be easier
        to just use the basic chat history rather than connect it to everything else.
    '''
    model: BaseChatModel
    history: MessageHistory = dataclasses.field(default_factory=MessageHistory)
    toolset: typing.Optional[ToolSet] = dataclasses.field(default_factory=None)

    @classmethod
    def from_openai(cls,
        model_name: str = "gpt-4o-mini", 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        tool_callable: typing.Optional[typing.Callable[[BaseChatModel],list[BaseTool]]] = None,
        **model_kwargs,
    ) -> typing.Self:
        '''Create a new chatbot with a chatgpt model.
        Args:
            model_name: name of the model to use
            system_prompt: first system message for the chat.
            tools: tools to be bound to the model using model.bind_tools(tools).
            tool_callable: function to get tools to use. Here so that the tools can access a reference to the model.
            model_kwargs: any additional arguments to pass to the model constructor.
        '''
        model = ChatOpenAI(
            model=model_name, 
            **model_kwargs
        )
        return cls.from_model(
            model = model,
            system_prompt = system_prompt,
            tools = cls._resolve_tools(model, tools, tool_callable),
        )

    @classmethod
    def from_ollama(cls,
        model_name: str = "llama3.1", 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        tool_callable: typing.Optional[typing.Callable[[BaseChatModel],list[BaseTool]]] = None,
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
            tools = cls._resolve_tools(model, tools, tool_callable),
        )
    
    @classmethod
    def _resolve_tools(cls, 
        model: BaseChatModel,
        tools: typing.Optional[list[BaseTool]], 
        tool_callable: typing.Optional[typing.Callable[[BaseChatModel],list[BaseTool]]],
    ) -> list[BaseTool]:
        '''Resolve the tools to use in the chatbot.'''
        if tool_callable is not None:
            if tools is None:
                return tool_callable(model)
            else:
                return tool_callable(model) + tools
        return tools
        

    @classmethod
    def from_model(cls,
        model: BaseChatModel, 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
    ) -> typing.Self:
        '''Create a new chatbot with any subtype of BaseChatModel.
        Args:
            model: chat model to use
            system_prompt: first system message
            tools: tools to be bound to the model using model.bind_tools(tools)
        '''
        
        if system_prompt is not None:
            history = MessageHistory.from_system_prompt(system_prompt)
        else:
            history = MessageHistory()

        if tools is not None:
            model = model.bind_tools(tools)
            toolset = ToolSet.from_list(tools)
        else:
            toolset = None
        
        return cls(
            model = model,
            history =  history,
            toolset = toolset,
        )

    def chat_stream(self, 
        new_message: typing.Optional[str], 
        tool_verbose_callback: typing.Callable[[str],None]|None = None,
    ) -> typing.Generator:
        '''Stream the chat. NOTE: CANNOT USE WITH TOOL CALLING! Need to raise better exception in the future.
        Description: calls model.stream() and yields the content of each chunk.
            I did this as a generator so I could append full response to end of message history.
            Mostly followed this tutorial:
                https://python.langchain.com/v0.1/docs/modules/model_io/llms/streaming_llm/
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will not be added to history..
            show_tools: whether to show tool calls in the response.
        '''
        if new_message is not None:
            self.history.add_human_message(new_message)
        
        full_message = None
        results = list()
        for chunk in self.model.stream(self.history.messages):            
            # concatenate chunks as it goes
            if full_message is None:
                full_message = chunk
            else:
                full_message += chunk

            # NOTE: tool calling doesn't really work in streaming, so not sure what to do?
            # now we handle any tool calls that happen
            results.append(self._handle_tool_calls(chunk, verbose_callback=tool_verbose_callback))
            
            # yield only the content to the user. Any metadata is stored in the history
            yield chunk.content

        self.history.add_message(full_message)

    def chat(self, 
        new_message: typing.Optional[str], 
        tool_verbose_callback: typing.Callable[[str],None]|None = None,
    ) -> str:
        '''Send a message to the chatbot and return the response.
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will not be added to history..
            show_tools: whether to show tool calls in the response.
        '''
        if new_message is not None:
            self.history.add_human_message(new_message)
        response = self.model.invoke(self.history.messages)
        self.history.add_message(response)
        results = self._handle_tool_calls(response, verbose_callback=tool_verbose_callback)
        if len(results) > 0:
            return self.chat(None)
        return response.content

    def _handle_tool_calls(self, 
        message: AIMessage, 
        verbose_callback: typing.Callable[[str],None]
    ) -> dict[str, ToolCallResult]:
        results = dict()
        for tool_info in message.tool_calls:
            #result, tool, tool_id = self.toolset.call_tool(tool_info, verbose=True)
            result = self.toolset.call_tool(tool_info, verbose_callback=verbose_callback)
            self.history.add_tool_message(result, result.id)
            results[result.tool.name] = result
        return results
        
    @property
    def ui(self) -> ChatBotUI:
        '''Expose diffferent UI for the chatbot.'''
        return ChatBotUI(self)
    
    