from __future__ import annotations

import typing
import dataclasses
import copy

import pydantic
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
    import pydantic


class UNSPECIFIED:
    pass

T = typing.TypeVar('T')
#StructuredOutput = pydantic.BaseModel # TODO: add other structured output types

@dataclasses.dataclass
class StructBot(typing.Generic[T]):
    '''A chatbot that uses a structured model and toolset.
    Attributes:
        _model: the chat model
        output_structure: the output structure
        history: the message history to send to the LLM.
    '''
    _model: BaseChatModel
    output_structure: typing.Type[T]
    history: MessageHistory = dataclasses.field(default_factory=MessageHistory)
    
    @classmethod
    def from_model(cls,
        model: BaseChatModel, 
        output_structure: typing.Type[T],
        system_prompt: str | None = None,
        history: MessageHistory | None = None,
    ) -> typing.Self:
        '''Create a new StructBot with any subtype of BaseChatModel.
        Args:
            model: chat model to use
            system_prompt: first system message
            tools: tools to be bound to the model using model.bind_tools(tools)
        '''
        if history is not None:
            if system_prompt is not None:
                raise ValueError("Cannot specify both system_prompt and history.")
            
            return cls(
                _model = model,
                output_structure = output_structure,
                history = history.clone(),
            )
        else:
            if system_prompt is not None:
                history = MessageHistory.from_system_prompt(system_prompt)
            else:
                history = MessageHistory()

            return cls(
                _model = model,
                output_structure = output_structure,
                history = history,
            )

    def invoke(
        self, 
        new_message: typing.Optional[str | HumanMessage], 
        add_to_history: bool = False,
        **kwargs,
    ) -> T:
        '''Invokes model and returns structured output type.'''
        messages = self._get_message_history(new_message, add_to_history)
        return self.model.invoke(messages, **kwargs)

    def _get_message_history(self, new_message: typing.Optional[str | HumanMessage], add_to_history: bool) -> list[BaseMessage]:
        '''Get messages for this chat and add the new message to the history if needed.'''
        if new_message is None:
            use_messages = self.history
        else:
            use_messages = self.history + [new_message]
            if add_to_history:
                self.history.add_human_message(new_message)
        return use_messages
    
    @property
    def model(self) -> BaseChatModel:
        '''Access the model with structured output.'''
        return self._model.with_structured_output(self.output_structure)


    ############################### Transformations and History Management ###############################
    def empty(self, keep_system_prompt: bool = False) -> typing.Self[T]:
        '''Returns a clone of the current chatbot with clean history. Does NOT modify history in-place!'''
        return self.clone(history=self.history.empty(keep_system_prompt=keep_system_prompt))

    def clone(   
        self, 
        output_structure: typing.Type[T] | None = None,
        model_transform: typing.Callable[[BaseChatModel],BaseChatModel] = lambda m: m,
        history: MessageHistory | None = None,
    ) -> StructBot[T]:
        '''Clone the StructBot with specified modifications..'''
        return self.__class__(
            _model = model_transform(self._model),
            output_structure = output_structure if output_structure is not None else self.output_structure,
            history = self.history if history is None else history,
        )





