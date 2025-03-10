from __future__ import annotations

import dataclasses
import typing
#import langchain_core.tools
from langchain_core.tools import BaseTool, BaseToolkit, render_text_description
from langchain_core.language_models import BaseChatModel

from .errors import ToolRaisedExceptionError, UknownToolError
from .types import ToolCallID, ToolName, UnspecifiedType, UNSPECIFIED
from .util import format_tool_text

if typing.TYPE_CHECKING:
    from .chatbot import ChatBot
    ToolFactoryType = typing.Callable[[ChatBot],list[BaseTool]]
    

@dataclasses.dataclass
class ToolSet:
    '''Class I made to manage tools. Create using .from_tools().
    Read more about making custom tools:
        https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/
    
    LangChain also includes a bunch of built-in tools. See them here:
        https://python.langchain.com/v0.2/docs/integrations/tools/
    '''
    tools: dict[ToolName, BaseTool]
    tool_factories: list[ToolFactoryType]
    tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None
    
    ################################# constructors #################################
    @classmethod
    def empty(cls) -> typing.Self:
        '''Shorthand for calling .from_tools with default argument values.'''
        return cls.from_tools()

    @classmethod
    def from_tools(cls, 
        tools: list[BaseTool] | None = None, 
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: list[ToolFactoryType] | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None = None,
    ) -> typing.Self:
        '''Create a toolset from a list of toolkits.
        Args:
            tools: list of tools to add before returning all tools
            toolkits: list of toolkits to add before returning all tools
            tool_factories: list of tool factories that will be called with ChatBot as an argument.
            tool_choice: tool to use when multiple tools are available
        '''
        return cls(
            tools = cls._tool_dict_from_lists(tools, toolkits),
            tool_factories = list(tool_factories) if tool_factories is not None else [],
            tool_choice = tool_choice,
        )
        
    ################################# accessing tools #################################
    def bind_tools(
        self, 
        chatbot: ChatBot | None = None,
    ) -> tuple[BaseChatModel, ToolLookup]:
        '''Create tools from factories and bind them to the model.'''
        tool_lookup = self.tool_lookup(chatbot=chatbot)
        if len(tool_lookup) > 0:
            if self.tool_choice is None:
                return chatbot._model.bind_tools(tool_lookup.tool_list()), tool_lookup
            else:
                return chatbot._model.bind_tools(tool_lookup.tool_list(), tool_choice=self.tool_choice), tool_lookup
        else:
            return chatbot._model, tool_lookup
        
    ################################# merging #################################
    def __add__(self, other: typing.Self) -> typing.Self:
        '''Merge two toolsets.'''
        return self.merge(other)
    
    def __union__(self, other: typing.Self) -> typing.Self:
        '''Merge two toolsets.'''
        return self.merge(other)

    def merge(self, other: typing.Self, replace_tool_choice: bool = False) -> typing.Self:
        '''Merge two toolsets.'''
        return self.__class__(
            tools = {**self.tools, **other.tools},
            tool_factories = self.tool_factories + other.tool_factories,
            tool_choice = other.tool_choice if replace_tool_choice else self.tool_choice,
        )

    def merge_tools(
        self,
        tools: list[BaseTool] | None = None,
        toolkits: list[BaseToolkit] | None = None,
        tool_factories: list[ToolFactoryType] | None = None,
        tool_choice: ToolName | typing.Literal['auto', 'any'] | None | UnspecifiedType = UNSPECIFIED,
    ) -> typing.Self:
        '''Merge the tools from the provided lists into a new toolset.'''
        return self.__class__(
            tools = {**self.tools, **self._tool_dict_from_lists(tools, toolkits)},
            tool_factories = list(self.tool_factories) + (list(tool_factories) if tool_factories is not None else []),
            tool_choice = tool_choice if tool_choice is not UNSPECIFIED else self.tool_choice,
        )
    
    ################################# accessing tool information #################################
    def tool_list(self, chatbot: ChatBot | None = None) -> list[BaseTool]:
        '''Get a list of tools.'''
        return list(self.tool_dict(chatbot=chatbot).values())
    
    def tool_lookup(self, chatbot: ChatBot | None = None) -> ToolLookup:
        '''Get a tool lookup.'''
        return ToolLookup.from_toolset(self, chatbot=chatbot)

    def tool_dict(self, chatbot: ChatBot | None = None) -> dict[ToolName, BaseTool]:
        '''Get a list of tools.'''
        if self.tool_factories is not None:
            if chatbot is None:
                raise ValueError('chatbot must be provided if tool factories are provided')
            factory_tools = {t.name: t for tf in self.tool_factories for t in tf(chatbot)}
        else:
            factory_tools = dict()

        toolname_overlap = set(self.tools.keys()) & set(factory_tools.keys())
        if len(toolname_overlap):
            raise ValueError('The following tools are defined in both tools and tool factories: ' + ', '.join(toolname_overlap))

        return {**self.tools, **factory_tools}

    @classmethod
    def _tool_dict_from_lists(
        cls,
        tools: list[BaseTool] | None = None, 
        toolkits: list[BaseToolkit] | None = None,
    ) -> dict[ToolName, BaseTool]:
        '''Get a dictionary of tools from a list of tools and toolkits.'''
        tools = list(tools) if tools is not None else []
        toolkits = list(toolkits) if toolkits is not None else []
        toolkit_tools = [tool for toolkit in toolkits for tool in toolkit.get_tools()]
        return {t.name:t for t in (tools + toolkit_tools)}

    ################################# cloning #################################
    def clone(self) -> typing.Self:
        '''Clone the toolset.'''
        return self.__class__(
            tools = dict(self.tools),
            tool_factories = list(self.tool_factories),
            tool_choice = self.tool_choice,
        )

    ################################# dunder #################################
    def __len__(self) -> int:
        '''Get the number of tools in the toolset.'''
        return len(self.tools) + len(self.tool_factories)



@dataclasses.dataclass
class ToolLookup:
    '''Maintains a dict of tool name -> tool mappings to be used at execution.
    '''
    tools: dict[ToolName, BaseTool]

    @classmethod
    def from_toolset(cls, toolset: ToolSet, chatbot: ChatBot | None = None) -> typing.Self:
        '''Create a tool lookup from a toolset.'''
        return cls(
            tools = toolset.tool_dict(chatbot=chatbot),
        )
    
    ################################# tool lookups #################################
    def get_tool_info(self, tool_info_dict: dict[str, str|dict]) -> ToolCallInfo:
        '''Get tool information and tool object reference.'''
        return ToolCallInfo.from_dict(tool_info=tool_info_dict, tool=self[tool_info_dict['name']])

    def __getitem__(self, name: str) -> BaseTool:
        '''Get a tool by name.'''
        try:
            return self.tools[name]
        except KeyError as e:
            raise UknownToolError.from_tool_name(
                tool_name = name,
                available_tools = self.names(),
            ) from e

    ################################# accessing other aspects of tools #################################
    def render(self) -> str:
        '''Gets description of toolset using the render method.'''
        return render_text_description(self.as_list())
    
    def names(self) -> list[ToolName]:
        '''Get the names of tools.'''
        return list(self.tools.keys())
    
    def tool_list(self) -> list[BaseTool]:
        return list(self.tools.values())

    ################################# dunder methods #################################
    def __len__(self) -> int:
        return len(self.tools)




@dataclasses.dataclass
class ToolCallInfo:
    '''Objectified verison of the tool information dict.'''
    id: ToolCallID
    name: str
    type: str
    args: dict[str, str|int|bool|float|list|dict]
    tool_call_args: dict[str, str|int|bool|float|list|dict] = dataclasses.field(repr=False)
    tool: BaseTool

    @classmethod
    def from_dict(cls, 
        tool_info: dict[str, str|dict],
        tool: BaseTool,
    ) -> typing.Self:
        '''Create a tool call result from tool info, tool, and return value.'''
        return cls(
            id = tool_info['id'],
            name = tool_info['name'],
            type = tool_info['type'],
            args = tool_info['args'],
            tool_call_args = tool_info,
            tool = tool,
        )
    
    def tool_info_str(self) -> str:
        '''Get tool call as a string.'''
        return format_tool_text(self.tool_call_args)
    

    def execute(self, chatbot: ChatBot|None = None, add_to_history: bool = True) -> ToolCallResult:
        '''Execute the tool call and return the result.'''
        try:
            return_value = self.tool.invoke(self.args)
        except Exception as e:
            raise ToolRaisedExceptionError.from_exception(self, e) from e
        
        result = ToolCallResult.from_tool_info(
            info = self, 
            return_value = return_value, 
        )

        if add_to_history:
            if chatbot is None:
                raise ValueError('chatbot must be provided if add_to_history is True')
            chatbot.history.add_tool_message(result.return_value, result.id)

        return result


@dataclasses.dataclass
class ToolCallResult:
    '''Result of a tool call.'''
    info: ToolCallInfo
    return_value: typing.Any

    @classmethod
    def from_tool_info(cls, 
        info: ToolCallInfo,
        return_value: typing.Any,
    ) -> typing.Self:
        '''Create a tool call result from tool info, tool, and return value.'''
        return cls(
            info = info,
            return_value = return_value,
        )

    @property
    def id(self) -> ToolCallID:
        return self.info.id
    
    @property
    def tool(self) -> BaseTool:
        return self.info.tool



