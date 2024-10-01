from __future__ import annotations

import typing
import dataclasses


from .message_history import AIMessage, AIMessageChunk

if typing.TYPE_CHECKING:
    from .chatbot import ChatBot
    from .toolset import ToolCallResult

class ChatResultBase:
    '''Base class for chat results.'''

    def _handle_tool_calls(self, 
        chatbot: ChatBot,
        message: AIMessage,
        add_to_history: bool = True,
    ) -> dict[str, ToolCallResult]:
        '''Handle the actual calling of tools.'''
        results: dict[str,ToolCallResult] = dict()
        for tool_info in message.tool_calls:
            result = chatbot.toolset.call_tool(tool_info)
            results[result.tool.name] = result

            if add_to_history:
                chatbot.history.add_tool_message(result.return_value, result.id)

            #if verbose:
            #    print(f'{result.tool_info_str} -> {result.return_value}')
        
        return results


@dataclasses.dataclass
class ChatResult(ChatResultBase):
    '''AI reply and results of any tool calls.'''
    chatbot: ChatBot
    message: AIMessage

    def call_tools(self, 
        add_to_history: bool = True, 
    ) -> dict[str, ToolCallResult]:
        '''Call tools on the full message.'''
        return self._handle_tool_calls(
            chatbot=self.chatbot, 
            message=self.message, 
            add_to_history=add_to_history, 
        )



@dataclasses.dataclass
class ChatStream(ChatResultBase):
    '''Returned from chat_stream so that user can collect results of streamed chat and tool calls.'''
    chatbot: ChatBot
    message_iter: typing.Iterator[AIMessageChunk]
    add_reply_to_history: bool = True
    full_message: AIMessage = dataclasses.field(default_factory=lambda: AIMessageChunk(content=''))
    exhausted: bool = False

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
    
    def call_tools(self, 
        add_to_history: bool = True, 
    ) -> dict[str, ToolCallResult]:
        '''Call tools on the full message.'''
        if not self.exhausted:
            raise ValueError('Cannot call tools until the stream is exhausted.')

        return self._handle_tool_calls(
            chatbot=self.chatbot, 
            message=self.full_message, 
            add_to_history=add_to_history, 
        )
    
