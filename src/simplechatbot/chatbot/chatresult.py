from __future__ import annotations

import typing
import dataclasses
import pydantic

from .message_history import AIMessage, AIMessageChunk

if typing.TYPE_CHECKING:
    from .chatbot import ChatBot

from .toolset import ToolCallInfo, ToolCallResult, ToolLookup



class ChatResultBase:
    '''Base class for chat results.'''

    @staticmethod
    def _handle_tool_calls(
        chatbot: ChatBot,
        tool_lookup: ToolLookup,
        message: AIMessage,
        add_to_history: bool,
    ) -> dict[str, ToolCallResult]:
        '''Actually execute tool calls and add results to history if requested.'''
        results: dict[str,ToolCallResult] = dict()
        for tool_info_dict in message.tool_calls:
            tool_info = tool_lookup.get_tool_info(tool_info_dict)
            result = tool_info.execute(chatbot, add_to_history=add_to_history)
            results[tool_info.name] = result
        
        return results

@dataclasses.dataclass(repr=False)
class ChatResult(ChatResultBase):
    '''AI reply and results of any tool calls.'''
    message: AIMessage
    chatbot: ChatBot
    tool_lookup: ToolLookup
    add_tool_calls_to_history: bool

    @classmethod
    def from_message(
        cls,
        message: AIMessage,
        chatbot: ChatBot,
        tool_lookup: ToolLookup,
        add_reply_to_history: bool,
        add_tool_calls_to_history: bool
    ) -> typing.Self:
        '''Create a chat stream from a message iterator.'''
        if add_reply_to_history:
            chatbot.history.add_message(message)

        return cls(
            message=message,
            chatbot=chatbot,
            tool_lookup = tool_lookup,
            add_tool_calls_to_history=add_tool_calls_to_history,
        )


    def execute_tools(self, 
    ) -> dict[str, ToolCallResult]:
        '''Call tools on the full message.'''
        return self._handle_tool_calls(
            chatbot=self.chatbot, 
            tool_lookup = self.tool_lookup,
            message=self.message, 
            add_to_history=self.add_tool_calls_to_history, 
        )
    
    @property
    def tool_calls(self) -> list[ToolCallInfo]:
        '''Get the names of the tools called.'''
        return [self.tool_lookup.get_tool_info(tc) for tc in self.message.tool_calls]
    
    @property
    def content(self) -> str:
        '''Get the content of the message.'''
        return self.message.content
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(content={self.content}, tool_calls={self.tool_calls})'
    
    def has_tool_calls(self) -> bool:
        '''Return whether the message has tool calls.'''
        return len(self.message.tool_calls) > 0



@dataclasses.dataclass
class StreamResult(ChatResultBase):
    '''Returned from chat_stream so that user can collect results of streamed chat and tool calls.'''
    message_iter: typing.Iterator[AIMessageChunk]
    chatbot: ChatBot
    tool_lookup: ToolLookup
    add_reply_to_history: bool
    #add_tool_calls_to_history: bool
    full_message: AIMessage
    exhausted: bool
    receive_callback: typing.Callable[[AIMessageChunk], None]

    @classmethod
    def from_message_iter(
        cls,
        message_iter: typing.Iterator[AIMessageChunk],
        chatbot: ChatBot,
        tool_lookup: ToolLookup,
        add_reply_to_history: bool,
        #add_tool_calls_to_history: bool,
        receive_callback: typing.Callable[[AIMessageChunk], None] = None,
    ) -> typing.Self:
        '''Create a chat stream from a message iterator.'''
        return cls(
            message_iter=message_iter,
            chatbot=chatbot,
            tool_lookup = tool_lookup,
            add_reply_to_history=add_reply_to_history,
            #add_tool_calls_to_history = add_tool_calls_to_history,
            full_message = AIMessageChunk(content=''),
            exhausted = False,
            receive_callback = receive_callback,
        )
    
    ####################### Iterating through results #######################
    def print_and_collect(self) -> ChatResult:
        '''Print the chat stream result and collect it.
        Example:
            result = agent.stream('hello world').print_and_collect()
            result.execute_tools()
        '''
        for chunk in self:
            print(chunk.content, end='', flush=True)
        return self.collect()

    def collect(self) -> ChatResult:
        '''Get the full chat result after accumulating all messages.'''
        if not self.exhausted:
            #raise ValueError('Cannot get chat result until the stream is exhausted.')
            for _ in self:
                pass

        return ChatResult.from_message(
            message=self.full_message,
            chatbot=self.chatbot,
            tool_lookup=self.tool_lookup,
            add_reply_to_history=False,
            add_tool_calls_to_history=self.add_reply_to_history,
        )

    def __iter__(self):
        return self

    def __next__(self):
        '''Get the next message and add it to the full message.'''
        try:
            next_message = next(self.message_iter)
            if self.receive_callback is not None:
                self.receive_callback(next_message)
            self.full_message += next_message
            return next_message
        
        except StopIteration:
            if self.add_reply_to_history:
                self.chatbot.history.add_message(self.full_message)
            self.exhausted = True
            raise StopIteration
    
    ####################### handle tool calls #######################
    def execute_tools(self, 
    ) -> dict[str, ToolCallResult]:
        '''Call tools on the full message.'''
        if not self.exhausted:
            raise ValueError('Cannot call tools until the stream is exhausted.')

        return self._handle_tool_calls(
            chatbot=self.chatbot, 
            tool_lookup = self.tool_lookup,
            message=self.full_message, 
            add_to_history=self.add_reply_to_history, 
        )

    @property
    def tool_calls(self) -> list[ToolCallInfo]:
        '''Get the names of the tools called.'''
        if not self.exhausted:
            raise ValueError('Cannot get tool calls until the stream is exhausted.')
        return [self.tool_lookup.get_tool_info(tc) for tc in self.full_message.tool_calls]
    
    def has_tool_calls(self) -> bool:
        '''Return whether the message has tool calls.'''
        return len(self.full_message.tool_calls) > 0
    

T = typing.TypeVar('T', bound=pydantic.BaseModel)

@dataclasses.dataclass(repr=False)
class StructuredOutputResult:
    '''Result of a structured output model. Use .data to access the result data.'''
    data: T
    chatbot: ChatBot
    add_reply_to_history: bool

    @classmethod
    def from_output(
        cls,
        output: T,
        chatbot: ChatBot,
        add_reply_to_history: bool,
    ) -> typing.Self:
        '''Create a chat stream from a message iterator.'''
        if add_reply_to_history:
            chatbot.history.add_ai_message(output.model_dump_json())

        return cls(
            data=output,
            chatbot=chatbot,
            add_reply_to_history=add_reply_to_history,
        )

    def as_json(self) -> str:
        '''Return the data as a json string.'''
        return self.data.model_dump_json()
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(data={self.data})'
    