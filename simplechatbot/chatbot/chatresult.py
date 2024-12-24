from __future__ import annotations

import typing
import dataclasses

from .message_history import AIMessage, AIMessageChunk

if typing.TYPE_CHECKING:
    from .chatbot import ChatBot

from .toolset import ToolCallInfo, ToolCallResult, ToolSet

class ChatResultBase:
    '''Base class for chat results.'''

    def _handle_tool_calls(self, 
        chatbot: ChatBot,
        toolset: ToolSet,
        message: AIMessage,
        add_reply_to_history: bool = True,
    ) -> dict[str, ToolCallResult]:
        '''Handle the actual calling of tools.'''
        results: dict[str,ToolCallResult] = dict()
        for tool_info_dict in message.tool_calls:
            tool_info = toolset.get_tool_info(tool_info_dict)
            result = tool_info.execute()

            if add_reply_to_history:
                chatbot.history.add_tool_message(result.return_value, result.id)

            #if verbose:
            #    print(f'{result.tool_info_str} -> {result.return_value}')
            results[tool_info.name] = result
        
        return results


@dataclasses.dataclass(repr=False)
class ChatResult(ChatResultBase):
    '''AI reply and results of any tool calls.'''
    message: AIMessage
    chatbot: ChatBot
    toolset: ToolSet
    add_reply_to_history: bool

    @classmethod
    def from_message(
        cls,
        message: AIMessage,
        chatbot: ChatBot,
        toolset: ToolSet,
        add_reply_to_history: bool,
    ) -> ChatStream:
        '''Create a chat stream from a message iterator.'''
        if add_reply_to_history:
            chatbot.history.add_message(message)

        return cls(
            message=message,
            chatbot=chatbot,
            toolset = toolset,
            add_reply_to_history=add_reply_to_history,
        )


    def execute_tools(self, 
    ) -> dict[str, ToolCallResult]:
        '''Call tools on the full message.'''
        return self._handle_tool_calls(
            chatbot=self.chatbot, 
            toolset = self.toolset,
            message=self.message, 
            add_reply_to_history=self.add_reply_to_history, 
        )
    
    @property
    def tool_calls(self) -> list[ToolCallInfo]:
        '''Get the names of the tools called.'''
        return [self.toolset.get_tool_info(tc) for tc in self.message.tool_calls]
    
    @property
    def content(self) -> str:
        '''Get the content of the message.'''
        return self.message.content
    
    def __repr__(self) -> str:
        return f'ChatResult(content={self.content}, tool_calls={self.tool_calls})'
    
    def has_tool_calls(self) -> bool:
        '''Return whether the message has tool calls.'''
        return len(self.message.tool_calls) > 0



@dataclasses.dataclass
class ChatStream(ChatResultBase):
    '''Returned from chat_stream so that user can collect results of streamed chat and tool calls.'''
    message_iter: typing.Iterator[AIMessageChunk]
    chatbot: ChatBot
    toolset: ToolSet
    add_reply_to_history: bool
    full_message: AIMessage
    exhausted: bool

    @classmethod
    def from_message_iter(
        cls,
        message_iter: typing.Iterator[AIMessageChunk],
        chatbot: ChatBot,
        toolset: ToolSet,
        add_reply_to_history: bool,
    ) -> ChatStream:
        '''Create a chat stream from a message iterator.'''
        return cls(
            message_iter=message_iter,
            chatbot=chatbot,
            toolset = toolset,
            add_reply_to_history=add_reply_to_history,
            full_message = AIMessageChunk(content=''),
            exhausted = False,
        )

    def __iter__(self):
        return self

    def __next__(self):
        '''Get the next message and add it to the full message.'''
        try:
            next_message = next(self.message_iter)
            self.full_message += next_message
            return next_message
        
        except StopIteration:
            if self.add_reply_to_history:
                self.chatbot.history.add_message(self.full_message)
            self.exhausted = True
            raise StopIteration
    
    def execute_tools(self, 
    ) -> dict[str, ToolCallResult]:
        '''Call tools on the full message.'''
        if not self.exhausted:
            raise ValueError('Cannot call tools until the stream is exhausted.')

        return self._handle_tool_calls(
            chatbot=self.chatbot, 
            toolset = self.toolset,
            message=self.full_message, 
            add_reply_to_history=self.add_reply_to_history, 
        )
    
    @property
    def tool_calls(self) -> list[ToolCallInfo]:
        '''Get the names of the tools called.'''
        if not self.exhausted:
            raise ValueError('Cannot get tool calls until the stream is exhausted.')
        return [self.toolset.get_tool_info(tc) for tc in self.full_message.tool_calls]
    
    def has_tool_calls(self) -> bool:
        '''Return whether the message has tool calls.'''
        return len(self.full_message.tool_calls) > 0
    
    def result(self) -> ChatResult:
        '''Get the full chat result after accumulating all messages.'''
        if not self.exhausted:
            #raise ValueError('Cannot get chat result until the stream is exhausted.')
            for _ in self:
                pass

        return ChatResult.from_message(
            message=self.full_message,
            chatbot=self.chatbot,
            toolset=self.toolset,
            add_reply_to_history=self.add_reply_to_history
        )
    
