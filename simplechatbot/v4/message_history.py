import typing
import dataclasses

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)

@dataclasses.dataclass
class MessageHistory:
    '''Maintains message history.
    LangChain actuall does provide convenient classes for this, but I found it easier to create my own.
        It's pretty much just a list of System/AI/Human messages used to generate model predictions.
        Full interface over messsage types from langchain_core.messages so the client doesn't need to use them.
    '''
    messages: list[BaseMessage] = dataclasses.field(default_factory=list)
    
    ############################# Constructors #############################
    @classmethod
    def from_system_prompt(cls, system_prompt: str) -> typing.Self:
        o = cls()
        o.add_system_message(system_prompt)
        return o

    ############################# Dunder #############################
    def __getitem__(self, key: int | slice) -> BaseMessage:
        return self.messages[key]
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __iter__(self) -> typing.Iterator:
        return iter(self.messages)
    
    ############################# Transformations #############################
    def get_buffer_string(self, *args, **kwargs) -> str:
        '''Get entire buffer as a string.'''
        return get_buffer_string(self.messages, *args, **kwargs)
    
    def render_streamlit(self, streamlit: typing.Any) -> str:
        '''Render the history in a streamlit friendly way.'''
        # Render the chat history.
        for msg in self.messages:
            streamlit.chat_message(msg.type).write(msg.content)
        return self.get_buffer_string()

    ############################# Accessors #############################
    @property
    def first_system(self) -> SystemMessage:
        '''Get the first system message.'''
        return self._first_of_type(self[:], SystemMessage)
    
    @property
    def last_ai(self) -> HumanMessage:
        '''Get the most recent ai message on the history.'''
        return self._first_of_type(self[::-1], AIMessage)

    @property
    def last_human(self) -> HumanMessage:
        '''Get the most recent human message on the history.'''
        return self._first_of_type(self[::-1], HumanMessage)

    @property
    def last(self) -> BaseMessage:
        '''Get the most recent message on the history.'''
        return self[-1]
    
    def _first_of_type(self, 
        message_iterable: typing.Iterable[BaseMessage],
        MessageType: typing.Type[BaseMessage],
    ) -> BaseMessage:
        for m in message_iterable:
            if isinstance(m, MessageType):
                return m
        
        # I use this exception type bc it is same used when x not found in list.index(x)
        raise ValueError(f'There are no messages of type {MessageType.__name__} in the history.')

    ############################# Adding Messages #############################
    def add_ai_chunks(self, chunks: list[AIMessageChunk]) -> None:
        '''Add a AIMessage to the history.
            Looks like they implemented the __add__ method to chunks so I'll just use that.
        '''
        self.messages.append(sum(chunks[1:], start=chunks[0]))

    def add_ai_message(self, content: str) -> None:
        '''Add a AIMessage to the history.'''
        self.messages.append(AIMessage(content=content))

    def add_system_message(self, content: str) -> None:
        '''Add a SystemMessage to the history.'''
        self.messages.append(SystemMessage(content=content))

    def add_human_message(self, content: str) -> None:
        '''Add a HumanMessage to the history.'''
        self.messages.append(HumanMessage(content=content))

    def add_tool_message(self, content: str, tool_call_id: str) -> None:
        '''Add a ToolMessage to the history.'''
        self.messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
    
    def add_message(self, message: BaseMessage) -> None:
        '''Add any subtype of BaseMessage to the history.'''
        self.messages.append(message)



