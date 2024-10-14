from __future__ import annotations

import typing
import dataclasses


# BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .message_history import MessageHistory
from .toolset import ToolSet, ToolCallResult

from .ui import ChatBotUI
from .chatresult import ChatResult, ChatStream

from langchain_core.messages import AIMessageChunk, AIMessage, BaseMessage, HumanMessage

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool, BaseToolkit


@dataclasses.dataclass
class ChatBot:
    '''Stores a chat model (or runnable interface), tools, and chat history.
    Created mostly from following this tutorial:
        https://python.langchain.com/v0.2/docs/tutorials/chatbot/

    Note that I'm only using a subset of the features there since I thought it'd be easier
        to just use the basic chat history rather than connect it to everything else.
    '''
    model: BaseChatModel
    history: MessageHistory = dataclasses.field(default_factory=MessageHistory)
    toolset: ToolSet = dataclasses.field(default_factory=ToolSet)

    ############################# Model-specific Constructors #############################
    @classmethod
    def from_openai(cls,
        model_name: str = "gpt-4o-mini", 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_constructor: typing.Optional[typing.Callable[[BaseChatModel],list[BaseTool]]] = None,
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
            tools = tools,
            toolkits = toolkits,
            tool_constructor = tool_constructor,
        )

    @classmethod
    def from_ollama(cls,
        model_name: str = "llama3.1", 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_constructor: typing.Optional[typing.Callable[[BaseChatModel],list[BaseTool]]] = None,
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
            tool_constructor = tool_constructor,
        )
    
        
    ############################# Generic Constructors #############################
    @classmethod
    def from_model(cls,
        model: BaseChatModel, 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_constructor: typing.Optional[typing.Callable[[BaseChatModel],list[BaseTool]]] = None,
    ) -> typing.Self:
        '''Create a new chatbot with any subtype of BaseChatModel.
        Args:
            model: chat model to use
            system_prompt: first system message
            tools: tools to be bound to the model using model.bind_tools(tools)
        '''
        
        # I'd like to think my MessageHistory does somethign that the langchain MessageHistory
        #   does not, but really it was just because I didn't know it was a thing before I implemented
        #   it. I'm not sure if I should keep it or not.
        if system_prompt is not None:
            history = MessageHistory.from_system_prompt(system_prompt)
        else:
            history = MessageHistory()

        # make toolset from toolkits and tools and tool_callable
        toolset = ToolSet.from_tools(
            model = model,
            tools = tools,
            toolkits = toolkits,
            tool_constructor = tool_constructor,
        )
        if len(toolset) > 0:
            model = model.bind_tools(toolset.get_tools())
        
        return cls(
            model = model,
            history = history,
            toolset = toolset,
        )

    ############################# Chat interface #############################
    def chat_stream(self, 
        new_message: typing.Optional[str], 
        add_to_history: bool = True,
    ) -> ChatStream:
        '''Return a ChatStream that can be iterated over to get the chat messages.
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will not be added to history.
        '''
        use_messages = self._handle_new_message(new_message, add_to_history)

        # result won't be ready until the stream is iterated over
        return ChatStream(
            chatbot = self,
            message_iter = self.model.stream(use_messages),
            add_reply_to_history = add_to_history,
        )

    def chat(self, 
        new_message: typing.Optional[str], 
        add_to_history: bool = True,
    ) -> ChatResult:
        '''Send a message to the chatbot and return the response.
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will not be added to history..
            show_tools: whether to show tool calls in the response.
            add_to_history: whether to add the message to the history after the response is received.
        '''
        use_messages = self._handle_new_message(new_message, add_to_history)
        result: AIMessage = self.model.invoke(use_messages)
        chatresult = ChatResult(
            chatbot = self,
            message = result,
        )

        # result is already available in this case, so just add to history now
        if add_to_history:
            self.history.add_message(result)

        return chatresult
        
        # call the model
        response = self.model.invoke(self.history + new_messages)

        # add new message and response to history
        if new_message is not None and add_to_history:
            self.history.add_human_message(new_message)
        self.history.add_message(response)

        # handle any tool calls that happened. if tools were called, follow up with new chat
        results = self._handle_tool_calls(response, result_callback=lambda: self.chat(None))        
        return response.content
    
    def _handle_new_message(self, new_message: str | HumanMessage, add_to_history: bool) -> list[BaseMessage]:
        '''Get messages for this chat and add the new message to the history if needed.'''
        if new_message is None:
            use_messages = self.history
        else:
            use_messages = self.history + [new_message]
            if add_to_history:
                self.history.add_human_message(new_message)
        return use_messages
    
    def _handle_tool_calls(self, 
        message: AIMessage, 
        result_callback: typing.Callable[[],str | typing.Generator[str]],
        verbose: bool = True,
    ) -> dict[str, ToolCallResult]:
        results: dict[str,ToolCallResult] = dict()
        for tool_info in message.tool_calls:
            result = self.toolset.call_tool(tool_info)
            self.history.add_tool_message(result.return_value, result.id)
            results[result.tool.name] = result
        
        if len(results) > 0:
            if verbose:
                for result in results.values():
                    print(f'{result.tool_info_str} -> {result.return_value}')
            return result_callback()

        return results

    ############################# wrappers over model calls #############################
    def stream(self, *args, **kwargs) -> typing.Iterator[AIMessageChunk]:
        '''Wrapper for model.stream.'''
        return self.model.stream(*args, **kwargs)
    
    def invoke(self, *args, **kwargs) -> AIMessage:
        '''Wrapper for model.invoke.'''
        return self.model.invoke(*args, **kwargs)
    
    ############################# method classes #############################
    @property
    def ui(self) -> ChatBotUI:
        '''Expose diffferent UI for the chatbot.'''
        return ChatBotUI(self)
        
    ############################# dunder #############################
    def __repr__(self) -> str:
        model_name = getattr(self.model, 'model_name', 'Unknown')
        return f'{self.__class__.__name__}(model_type={type(self.model).__name__}, model_name="{model_name}", tools={self.toolset.names() if len(self.toolset) else None})'

