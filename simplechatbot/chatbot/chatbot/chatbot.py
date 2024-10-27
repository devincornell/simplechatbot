from __future__ import annotations

import typing
import dataclasses


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
            tools = tools,
            toolkits = toolkits,
        )

        return cls(
            _model = model,
            history = history,
            toolset = toolset,
        )

    ############################# Chat interface #############################
    def chat_stream(self, 
        new_message: typing.Optional[str], 
        add_to_history: bool = True,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
    ) -> ChatStream:
        '''Return a ChatStream that can be iterated over to get the chat messages.
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will not be added to history.
        '''
        use_messages = self._handle_new_message(new_message, add_to_history)

        # result won't be ready until the stream is iterated over
        return ChatStream(
            chatbot = self,
            message_iter = self.get_tool_model(tools=tools, toolkits=toolkits).stream(use_messages),
            add_reply_to_history = add_to_history,
        )

    def chat(self, 
        new_message: typing.Optional[str], 
        add_to_history: bool = True,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
    ) -> ChatResult:
        '''Send a message to the chatbot and return the response.
        Args:
            new_message: message to send to the chatbot. If None is entered, a new message will not be added to history..
            show_tools: whether to show tool calls in the response.
            add_to_history: whether to add the message to the history after the response is received.
        '''
        use_messages = self._handle_new_message(new_message, add_to_history)
        result: AIMessage = self.get_tool_model(tools=tools, toolkits=toolkits).invoke(use_messages)
        chatresult = ChatResult(
            chatbot = self,
            message = result,
        )

        # result is already available in this case, so just add to history now
        if add_to_history:
            self.history.add_message(result)

        return chatresult
    
    def _handle_new_message(self, new_message: typing.Optional[str | HumanMessage], add_to_history: bool) -> list[BaseMessage]:
        '''Get messages for this chat and add the new message to the history if needed.'''
        if new_message is None:
            use_messages = self.history
        else:
            use_messages = self.history + [new_message]
            if add_to_history:
                self.history.add_human_message(new_message)
        return use_messages
    

    ############################# wrappers over model calls #############################
    def stream(self, *args, **kwargs) -> typing.Iterator[AIMessageChunk]:
        '''Wrapper for model.stream.'''
        return self.get_tool_model().stream(*args, **kwargs)
    
    def invoke(self, *args, **kwargs) -> AIMessage:
        '''Wrapper for model.invoke.'''
        return self.get_tool_model().invoke(*args, **kwargs)
    
    ############################# method classes #############################
    @property
    def model(self) -> BaseChatModel:
        return self.get_tool_model()
    
    def get_tool_model(self,
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
    ) -> BaseChatModel:
        '''Get the chat model bound by tools.'''
        toolset = self.toolset | ToolSet.from_tools(tools=tools, toolkits=toolkits)
        if len(toolset) > 0:
            return self._model.bind_tools(toolset.get_tools())
        else:
            return self._model
    
    ############################# method classes #############################
    @property
    def ui(self) -> ChatBotUI:
        '''Expose diffferent UI for the chatbot.'''
        return ChatBotUI(self)
        
    ############################# dunder #############################
    def __repr__(self) -> str:
        model_name = getattr(self._model, 'model_name', 'Unknown')
        return f'{self.__class__.__name__}(model_type={type(self._model).__name__}, model_name="{model_name}", tools={self.toolset.names() if len(self.toolset) else None})'

