from __future__ import annotations

import dataclasses
import typing
#import langchain_core.tools
from langchain_core.tools import BaseTool, BaseToolkit, render_text_description
from langchain_core.language_models import BaseChatModel

from .errors import ToolRaisedExceptionError, UknownToolError
from .types import ToolCallID
from .util import format_tool_text


@dataclasses.dataclass
class ToolCallResult:
    '''Result of a tool call.'''
    id: ToolCallID
    name: str
    type: str
    args: dict[str, str|int|bool|float|list|dict]
    tool: BaseTool
    return_value: typing.Any
    tool_call_args: dict[str, str|int|bool|float|list|dict] = dataclasses.field(repr=False)

    @classmethod
    def from_tool_info(cls, 
        tool_info: dict[str, str|dict],
        tool: BaseTool,
        return_value: typing.Any,
    ) -> typing.Self:
        '''Create a tool call result from tool info, tool, and return value.'''
        return cls(
            id = tool_info['id'],
            name = tool_info['name'],
            type = tool_info['type'],
            args = tool_info['args'],
            tool_call_args = tool_info,
            tool = tool,
            return_value = return_value,
        )

    #@property
    #def id(self) -> ToolCallID:
    #    return self.tool_info['id']

    #@property
    #def tool_name(self) -> str:
    #    return self.tool.name
    
    def tool_info_str(self) -> str:
        '''Get tool call as a string.'''
        return format_tool_text(self.tool_call_args)
    

@dataclasses.dataclass
class ToolSet:
    '''Class I made to manage tools.
    Read more about making custom tools:
        https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/
    
    LangChain also includes a bunch of built-in tools. See them here:
        https://python.langchain.com/v0.2/docs/integrations/tools/
    '''
    tools: dict[str, BaseTool] = dataclasses.field(default_factory=dict)
        
    @classmethod
    def from_tools(cls, 
        tools: typing.Optional[list[BaseTool]] = None,
        toolkits: typing.Optional[list[BaseToolkit]] = None,
    ) -> typing.Self:
        '''Create toolset from a list of tools and toolkits.'''
        tools = list(tools) if tools is not None else []

        if toolkits is not None:
            toolkit_tools = [tool for toolkit in toolkits for tool in toolkit.get_tools()]
        else:
            toolkit_tools = []

        return cls(
            tools = {t.name:t for t in (tools + toolkit_tools)},
        )

    def call_tool(self, 
        tool_info: dict[str, str|dict], 
    ) -> ToolCallResult:
        '''Extracts tool information and executes tool.
        Args:
            tool_info is the dict returned from the chat model that includes tool name and arguments.
        '''
        tool = self[tool_info['name']] # NOTE: raises UnidentifiedToolError if tool isn't found
        args = tool_info['args']
        try:
            return_value = tool.invoke(args)
        except Exception as e:
            raise ToolRaisedExceptionError.from_exception(tool_info, tool, e) from e
        
        return ToolCallResult.from_tool_info(
            tool_info = tool_info, 
            tool = tool,
            return_value = return_value, 
        )

    def get_tools(self) -> list[BaseTool]:
        '''Get a list of tools.'''
        return list(self.tools.values())

    def render(self) -> str:
        '''Gets description of toolset using the render method.'''
        return render_text_description(self.as_list())
    
    def names(self) -> list[str]:
        return list(self.tools.keys())
    
    def merge(self, other: typing.Self) -> typing.Self:
        return ToolSet.from_tools(tools=self.get_tools() + other.get_tools())
    
    ##################### dunder methods #####################
    def __or__(self, other: typing.Self) -> typing.Self:
        return self.merge(other)

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
