from __future__ import annotations

import typing
import dataclasses
import copy

from langchain_core.messages import AIMessageChunk, AIMessage, BaseMessage, HumanMessage
import pydantic

from .message_history import MessageHistory
from .toolset import ToolSet, ToolCallResult, ToolLookup

from .ui import ChatBotUI
from .chatresult import ChatResult, ChatStream
from .structbot import StructBot

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool, BaseToolkit
    from .toolset import ToolFactoryType

from .types import ToolName, ToolCallID, UNSPECIFIED, UnspecifiedType



@dataclasses.dataclass
class ChatBot:
    '''Stores a chat model (or runnable interface), tools, and chat history.
    Created mostly from following this tutorial:
        https://python.langchain.com/v0.2/docs/tutorials/chatbot/

    Note that I'm only using a subset of the features there since I thought it'd be easier
        to just use the basic chat history rather than connect it to everything else.
    '''
    _model: BaseChatModel
    history: MessageHistory = dataclasses.field(default_factory=MessageHistory)
    toolset: ToolSet = dataclasses.field(default_factory=ToolSet)
    
    ############################# Generic Constructors #############################
    @classmethod
    def from_model(cls,
        model: BaseChatModel, 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None,
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

        # NOTE: ChatBot is in a partially initialized state here, so maybe fix that in the future.
        new_chatbot = cls(
            _model = model,
            history = history,
            toolset = ToolSet.from_tools(
                tools = tools,
                toolkits = toolkits,
                tool_factories = tool_factories,
                tool_choice=tool_choice,
            ),
        )
        return new_chatbot
    
    
    ############################# Chat interface #############################
    def stream(self, 
        new_message: typing.Optional[str], 
        add_to_history: bool = True,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_factories: ToolFactoryType | None = None,
        do_print: bool = False,
        receive_callback: typing.Callable[[AIMessageChunk], None] = None,
    ) -> ChatStream:
        '''Return a ChatStream that can be iterated over to get the chat messages.
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will 
                not be added to history.
            add_to_history: whether to add the message to the history after the response is received
            tools: tools to use in this particular message.
            toolkits: toolkits to use in this particular message.
            tool_factories: tool factories to use in this particular message.
        '''
        use_messages = self._get_message_history(new_message, add_to_history)
        return self._stream(
            messages = use_messages,
            add_reply_to_history=add_to_history,
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
            receive_callback = receive_callback,
            do_print=do_print,
        )

    def chat(self, 
        new_message: typing.Optional[str], 
        add_to_history: bool = True,
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: ToolFactoryType | None = None,
    ) -> ChatResult:
        '''Send a message to the chatbot and return the response.
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will not be added to history..
            show_tools: whether to show tool calls in the response.
            add_to_history: whether to add the message to the history after the response is received.
        '''
        use_messages = self._get_message_history(new_message, add_to_history=add_to_history)
        return self._invoke(
            messages = use_messages,
            add_reply_to_history=add_to_history,
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
        )
    
    def _get_message_history(self, new_message: typing.Optional[str | HumanMessage], add_to_history: bool) -> list[BaseMessage]:
        '''Get messages for this chat and add the new message to the history if needed.'''
        if new_message is None:
            use_messages = self.history
        else:
            use_messages = self.history + [new_message]
            if add_to_history:
                self.history.add_human_message(new_message)
        return use_messages
    
    ############################# wrappers over model calls #############################
    def _stream(
        self, 
        messages: BaseMessage | str | list[BaseMessage] | list[str],
        add_reply_to_history: bool = False,
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | UnspecifiedType | None = UNSPECIFIED,
        receive_callback: typing.Callable[[AIMessageChunk], None] = None,
        do_print: bool = False,
        **kwargs,
    ) -> ChatStream:
        '''Sends a message to be streamed back without storing the message as history.'''
        self.history.check_tools_were_executed()
        model, tool_lookup = self.get_model_with_tools(
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
            tool_choice=tool_choice,
        )

        if do_print and receive_callback is None:
            receive_callback = lambda r: print(r.content, end='', flush=True)
        elif do_print and receive_callback is not None:
            receive_callback = lambda r: (print(r.content, end='', flush=True), receive_callback(r))
        
        return ChatStream.from_message_iter(
            message_iter = model.stream(messages, **kwargs),
            chatbot = self,
            tool_lookup=tool_lookup,
            add_reply_to_history = add_reply_to_history,
            receive_callback=receive_callback,
        )

    def _invoke(
        self, 
        messages: BaseMessage | str | list[BaseMessage] | list[str],
        add_reply_to_history: bool = False,
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | UnspecifiedType | None = UNSPECIFIED,
        **kwargs,
    ) -> ChatResult:
        '''Invoke the model and return a chatresult object.'''
        self.history.check_tools_were_executed()
        model, tool_lookup = self.get_model_with_tools(
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
            tool_choice=tool_choice,
        )
        return ChatResult.from_message(
            message = model.invoke(messages, **kwargs),
            chatbot = self,
            tool_lookup=tool_lookup,
            add_reply_to_history = add_reply_to_history,
        )

    
    ############################# access model with tools #############################
    @property
    def model(self) -> BaseChatModel:
        '''Access the model with tools bound to it.'''
        m, tl = self.get_model_with_tools()
        return m

    def get_model_with_tools(
        self, 
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None | UnspecifiedType = UNSPECIFIED,
    ) -> tuple[BaseChatModel, ToolLookup]:
        '''Bind tools to the model and return the resulting chain.
        Args:
            tools: tools to bind to the model.
            toolkits: toolkits to bind to the model.
            tool_factories: tool factories to bind to the model.
            tool_choice: how to choose the tools to bind to the model.
        '''
        toolset = self.toolset.merge_tools(
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
            tool_choice=tool_choice,
        )

        model, tool_lookup = toolset.bind_tools(chatbot=self)
        
        return model, tool_lookup

    ############################# cloning #############################
    def fresh(
        self, 
        system_prompt: typing.Optional[str] = None,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None,
    ) -> typing.Self:
        '''Create a fresh chatbot using the model from this instance.
        Args:
            system_prompt: system prompt to use.
            tools: tools to bind to the model.
            toolkits: toolkits to bind to the model.
            tool_factories: tool factories to bind to the model.
            tool_choice: how to choose the tools to bind to the model.
        '''
        return self.from_model(
            model = self._model,
            system_prompt = system_prompt,
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
            tool_choice=tool_choice,
        )
    
    def empty(
        self, 
        keep_system_prompt: bool = False, 
        clear_tools: bool = True,
    ) -> typing.Self:
        '''Create an empty chatbot, keeping the system prompt if desired.
        Args:
            keep_system_prompt: whether to keep the system prompt.
            clear_tools: whether to clear the tools.
        '''
        return self.clone(
            history = self.history.empty(keep_system_prompt=keep_system_prompt),
            toolset=self.toolset.empty() if clear_tools else self.toolset.clone(),
        )

    def clone(
        self, 
        model_transform: typing.Callable[[BaseChatModel],BaseChatModel] = lambda m: m,
        system_prompt: str | None = None,
        history: MessageHistory | None = None,
        toolset: ToolSet | None = None,
    ) -> typing.Self:
        '''Clone this instance, keeping some aspects the same and changing others.
        Description: Makes a copy of this chatbot. If toolset or history is specified,
            a clone of them will replace the old instances. If system_prompt is specified, it will 
            create a new history with ONLY the system prompt. system_prompt cannot be used
            with history.

            If you want to completely clear history, use the empty() method first.

        Args:
            model_transform: function to create a new model from the old one. Use to bind tools, etc.
            system_prompt: if specified, will create a new history with only the system prompt.
            history: new history to use. If None, a clone of the old history is used.
            toolset: new toolset to use. If None, a clone of the old toolset is used.
        '''
        if system_prompt is not None:
            if history is not None:
                raise ValueError("Cannot specify both system_prompt and history.")
            else:
                history = MessageHistory.from_system_prompt(system_prompt)
        else:
            if history is not None:
                history = history.clone()
            else:
                history = self.history.clone()
            
        return self.__class__(
            _model = model_transform(self._model),
            history = history,
            toolset = toolset.clone() if toolset is not None else self.toolset.clone(),
        )

    ############################# method classes #############################
    def structbot_from_model(
        self, 
        output_structure: pydantic.BaseModel,
        system_prompt: str | None = None,
        history: MessageHistory | None = None,
    ) -> StructBot:
        '''Create a StructBot from the model in this chatbot.
        Args:
            output_structure: the output structure to use.
            system_prompt: the system prompt to use.
            history: history to add to structbot.
        '''

        return StructBot.from_model(
            model=self._model, 
            output_structure=output_structure,
            system_prompt=system_prompt,
            history = history,
        )
    
    @property
    def ui(self) -> ChatBotUI:
        '''Expose diffferent UI for the chatbot.'''
        return ChatBotUI(self)
        
    ############################# dunder #############################
    def __repr__(self) -> str:
        model_name = getattr(self._model, 'model_name', 'Unknown')
        tool_names = self.toolset.tool_lookup(chatbot=self)
        return f'{self.__class__.__name__}(model_type={type(self._model).__name__}, model_name="{model_name}", tools={tool_names})'

