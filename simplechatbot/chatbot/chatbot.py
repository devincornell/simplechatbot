from __future__ import annotations

import typing
import dataclasses
import copy

from langchain_core.messages import AIMessageChunk, AIMessage, BaseMessage, HumanMessage

from .message_history import MessageHistory
from .toolset import ToolSet, ToolCallResult

from .ui import ChatBotUI
from .chatresult import ChatResult, ChatStream


if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool, BaseToolkit
    from .toolset import ToolFactoryType
    from .types import ToolName, ToolCallID


class UNSPECIFIED:
    pass

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
    tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None
    
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
            toolset = ToolSet.empty(),
            tool_choice = tool_choice,
        )
        new_chatbot.toolset = ToolSet.from_tools(
            chatbot=new_chatbot,
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
        )

        return new_chatbot
    
    
    ############################# Chat interface #############################
    def chat_stream(self, 
        new_message: typing.Optional[str], 
        add_to_history: bool = True,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
        tool_factories: ToolFactoryType | None = None,
    ) -> ChatStream:
        '''Return a ChatStream that can be iterated over to get the chat messages.
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will not be added to history.
        '''
        use_messages = self._get_message_history(new_message, add_to_history)
        return self.stream(
            messages = use_messages,
            add_reply_to_history=add_to_history,
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
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
        return self.invoke(
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
                    print(f'{result.info.tool_info_str()} -> {result.return_value}')
            return result_callback()

        return results

    ############################# wrappers over model calls #############################
    def stream(
        self, 
        messages: BaseMessage | str | list[BaseMessage] | list[str],
        add_reply_to_history: bool = False,
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None,
        **kwargs,
    ) -> ChatStream:
        '''Wrapper for model.stream.'''
        self.history.check_tools_were_executed()
        model, toolset = self.get_model_with_tools(
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
            tool_choice=tool_choice,
        )
        
        return ChatStream.from_message_iter(
            message_iter = model.stream(messages, **kwargs),
            chatbot = self,
            toolset=toolset,
            add_reply_to_history = add_reply_to_history,
        )

    def invoke(
        self, 
        messages: BaseMessage | str | list[BaseMessage] | list[str],
        add_reply_to_history: bool = False,
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None,
        **kwargs,
    ) -> ChatResult:
        '''Wrapper for model.invoke.'''
        self.history.check_tools_were_executed()
        model, toolset = self.get_model_with_tools(
            tools = tools,
            toolkits = toolkits,
            tool_factories = tool_factories,
            tool_choice=tool_choice,
        )
        return ChatResult.from_message(
            message = model.invoke(messages, **kwargs),
            chatbot = self,
            toolset=toolset,
            add_reply_to_history = add_reply_to_history,
        )

    
    ############################# access model with tools #############################
    @property
    def model(self) -> BaseChatModel:
        '''Get the model with the tools bound to it.'''
        m, ts = self.get_model_with_tools()
        return m

    def get_model_with_tools(
        self, 
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None,
    ) -> tuple[BaseChatModel, ToolSet]:
        '''Bind tools to the model and return the resulting chain.'''
        tool_choice = tool_choice if tool_choice is not None else self.tool_choice
        toolset = self.toolset.merge_new_tools(
            tools = tools,
            toolkits = toolkits,
            chatbot = self,
            tool_factories = tool_factories,
        )

        model = toolset.bind_tools(self._model, tool_choice = tool_choice)
        
        return model, toolset

    ############################# cloning #############################    
    def empty(self, keep_system_prompt: bool = False, clear_tools: bool = True) -> typing.Self:
        '''Create an empty chatbot, keeping the system prompt if desired.'''
        return self.clone(
            clear_history=True,
            keep_system_prompt=keep_system_prompt,
            clear_tools=clear_tools,
        )

    def clone(
        self, 
        model_factory: typing.Callable[[BaseChatModel],BaseChatModel] = lambda m: m,
        clear_history: bool = False,
        keep_system_prompt: bool | typing.Type[UNSPECIFIED] = UNSPECIFIED,
        clear_tools: bool = False,
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: ToolFactoryType | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None | typing.Type[UNSPECIFIED] = UNSPECIFIED,
    ) -> typing.Self:
        '''Clone this instance, keeping some aspects the same and changing others.
        Args:
            model_factory: function to create a new model from the old one. Use to bind tools, etc.
            clear_history: whether to clear the history
            keep_system_prompt: whether to keep the system prompt
            clear_tools: whether to clear the tools
            tools: tools to add to the toolset
            toolkits: toolkits to add to the toolset
            tool_factories: tool factories to add to the toolset
            tool_choice: tool to use for the model
        '''
        if clear_history and keep_system_prompt is UNSPECIFIED:
            raise ValueError('keep_system_prompt must be specified if clear_history is True')
        if not clear_history and keep_system_prompt is not UNSPECIFIED:
            raise ValueError('keep_system_prompt can only be specified if clear_history is True')

        toolset = self.toolset.empty() if clear_tools else self.toolset.clone()
        toolset = toolset.merge_new_tools(
            tools = tools,
            toolkits = toolkits,
            chatbot = self,
            tool_factories = tool_factories,
        )
        return self.__class__(
            _model = model_factory(self._model),
            history = self.history.empty(keep_system_prompt=keep_system_prompt) if clear_history else self.history.clone(),
            toolset = toolset,
            tool_choice = tool_choice if tool_choice is not UNSPECIFIED else self.tool_choice,
        )


    ############################# method classes #############################
    @property
    def ui(self) -> ChatBotUI:
        '''Expose diffferent UI for the chatbot.'''
        return ChatBotUI(self)
        
    ############################# dunder #############################
    def __repr__(self) -> str:
        model_name = getattr(self._model, 'model_name', 'Unknown')
        return f'{self.__class__.__name__}(model_type={type(self._model).__name__}, model_name="{model_name}", tools={self.toolset.names() if len(self.toolset) else None})'

