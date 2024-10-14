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
    tool_info: dict[str, str|dict]
    tool: BaseTool
    return_value: typing.Any

    @property
    def id(self) -> ToolCallID:
        return self.tool_info['id']

    @property
    def tool_name(self) -> str:
        return self.tool.name
    
    @property
    def tool_info_str(self) -> str:
        '''Get tool call as a string.'''
        return format_tool_text(self.tool_info)
    

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
        model: BaseChatModel,
        tool_constructor: typing.Optional[typing.Callable[[BaseChatModel],list[BaseTool]]],
        tools: typing.Optional[list[BaseTool]], 
        toolkits: typing.Optional[list[BaseToolkit]],
    ) -> typing.Self:
        '''Create a toolset from a list of tools or a callable that returns a list of tools.'''
        tools = tools if tools is not None else []
        return cls.from_list(
            tool_list=tools + (tool_constructor(model) if tool_constructor is not None else []),
            toolkits = toolkits,
        )
    
    @classmethod
    def from_list(cls, 
        tool_list: typing.Optional[list[BaseTool]],
        toolkits: typing.Optional[list[BaseToolkit]],
    ) -> typing.Self:
        '''Create toolset from a list of tools and toolkits.'''
        tool_list = list(tool_list) if tool_list is not None else []

        if toolkits is not None:
            toolkit_tools = [tool for toolkit in toolkits for tool in toolkit.get_tools()]
        else:
            toolkit_tools = []

        return cls(
            tools = {t.name:t for t in tool_list + toolkit_tools},
        )

    def call_tool(self, 
        tool_info: dict[str, str|dict], 
    ) -> ToolCallResult:
        '''Extracts tool information and executes tool.'''
        tool = self[tool_info['name']] # NOTE: raises UnidentifiedToolError if tool isn't found
        args = tool_info['args']
        try:
            return_value = tool.invoke(args)
        except Exception as e:
            raise ToolRaisedExceptionError.from_exception(tool_info, tool, e) from e
        
        return ToolCallResult(
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
    
    ##################### dunder methods #####################
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
