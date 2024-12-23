from __future__ import annotations

import dataclasses
import typing
#import langchain_core.tools
from langchain_core.tools import BaseTool, BaseToolkit, render_text_description
from langchain_core.language_models import BaseChatModel

from .errors import ToolRaisedExceptionError, UknownToolError
from .types import ToolCallID, ToolName
from .util import format_tool_text

if typing.TYPE_CHECKING:
    from .chatbot import ChatBot
    ToolFactoryType = list[typing.Callable[[ChatBot],list[BaseTool]]]
    

@dataclasses.dataclass
class ToolSet:
    '''Class I made to manage tools. Create using .from_tools().
    Read more about making custom tools:
        https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/
    
    LangChain also includes a bunch of built-in tools. See them here:
        https://python.langchain.com/v0.2/docs/integrations/tools/
    '''
    tools: dict[ToolName, BaseTool]
    
    ################################# constructors #################################
    @classmethod
    def from_tools(cls, 
        tools: list[BaseTool] | None = None, 
        toolkits: list[BaseToolkit] | None = None,
        chatbot: ChatBot | None = None,
        tool_factories: ToolFactoryType | None = None,
    ) -> typing.Self:
        '''Create a toolset from a list of toolkits.
        Args:
            chatbot: chatbot from which to call tool factories
            tools: list of tools to add before returning all tools
            toolkits: list of toolkits to add before returning all tools
        '''
        tools = list(tools) if tools is not None else []

        toolkits = list(toolkits) if toolkits is not None else []
        if toolkits is not None:
            toolkit_tools = [tool for toolkit in toolkits for tool in toolkit.get_tools()]
        else:
            toolkit_tools = []

        tool_factories = list(tool_factories) if tool_factories is not None else []
        factory_tools = [tool for factory in tool_factories for tool in factory(chatbot)]

        return cls(
            tools = {t.name:t for t in (tools + toolkit_tools + factory_tools)},
        )
    
    @classmethod
    def empty(cls) -> typing.Self:
        '''Create an empty toolset.'''
        return cls(tools = {})

    ################################# function calling #################################
    #def call_tool(self, 
    #    tool_info: dict[str, str|dict], 
    #) -> ToolCallResult:
    #    '''Extracts tool information and executes tool.'''
    #    tool = self[tool_info['name']] # NOTE: raises UnidentifiedToolError if tool isn't found
    #    args = tool_info['args']
    #    try:
    #        return_value = tool.invoke(args)
    #    except Exception as e:
    #        raise ToolRaisedExceptionError.from_exception(tool_info, tool, e) from e
    #    
    #    return ToolCallResult.from_tool_info(
    #        tool_info = tool_info, 
    #        tool = tool,
    #        return_value = return_value, 
    #    )
    
    def get_tool_info(self, tool_info_dict: dict[str, str|dict]) -> ToolCallInfo:
        '''Get tool information and tool object reference.'''
        return ToolCallInfo.from_dict(tool_info=tool_info_dict, tool=self[tool_info_dict['name']])
    
    ################################# accessing tools #################################
    def bind_tools(self, model: BaseChatModel) -> BaseChatModel:
        '''Bind tools to a model.'''
        if len(self.tools) > 0:
            return model.bind_tools(self.tool_list())
        else:
            return model
        
    def tool_list(self) -> list[BaseTool]:
        '''Get a list of tools.'''
        return list(self.tools.values())
    
    def tool_dict(self) -> dict[ToolName,BaseTool]:
        '''Get a dictionary of tools.'''
        return dict(self.tools)

    ##################### merging toolsets #####################
    def merge_new_tools(
        self,
        tools: list[BaseTool] | None = None, 
        toolkits: list[BaseToolkit] | None = None,
        chatbot: ChatBot | None = None,
        tool_factories: ToolFactoryType | None = None,
    ) -> typing.Self:
        '''Merge new tools into the toolset and return a new instance.'''
        other = self.from_tools(
            tools = tools,
            toolkits = toolkits,
            chatbot = chatbot,
            tool_factories = tool_factories,
        )
        return self.merge(other)

    def __add__(self, other: typing.Self) -> typing.Self:
        '''Merge two toolsets.'''
        return self.merge(other)
    
    def __union__(self, other: typing.Self) -> typing.Self:
        '''Merge two toolsets.'''
        return self.merge(other)

    def merge(self, other: typing.Self) -> typing.Self:
        '''Merge two toolsets.'''
        return self.__class__(
            tools = {**self.tools, **other.tools},
        )

    ################################# accessing other aspects of tools #################################
    def render(self) -> str:
        '''Gets description of toolset using the render method.'''
        return render_text_description(self.as_list())
    
    def names(self) -> list[str]:
        return list(self.tools.keys())
    
    
    ################################# dunder methods #################################
    def __len__(self) -> int:
        return len(self.tools)
    
    def __getitem__(self, name: str) -> BaseTool:
        '''Get a tool by name.'''
        try:
            return self.tools[name]
        except KeyError as e:
            raise UknownToolError.from_tool_name(
                tool_name = name,
                available_tools = self.names(),
            ) from e



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
    

    def execute(self) -> ToolCallResult:
        '''Execute the tool call and return the result.'''
        try:
            return_value = self.tool.invoke(self.args)
        except Exception as e:
            raise ToolRaisedExceptionError.from_exception(self, e) from e
        
        return ToolCallResult.from_tool_info(
            tool_info = self, 
            return_value = return_value, 
        )



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
