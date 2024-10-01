from __future__ import annotations

import dataclasses
import typing
import langchain_core.tools

from .errors import ToolRaisedExceptionError, UknownToolError
from .types import ToolCallID
from .util import format_tool_text


@dataclasses.dataclass
class ToolCallResult:
    '''Result of a tool call.'''
    tool_info: dict[str, str|dict]
    tool: langchain_core.tools.BaseTool
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
    tools: dict[str, langchain_core.tools.BaseTool]

    @classmethod
    def from_list(cls, tool_list: list[langchain_core.tools.BaseTool]) -> typing.Self:
        return cls(
            tools = {t.name:t for t in tool_list},
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

    def __getitem__(self, name: str) -> langchain_core.tools.BaseTool:
        '''Get a tool by name.'''
        try:
            return self.tools[name]
        except KeyError as e:
            raise UknownToolError.from_tool_name(
                tool_name = name,
                available_tools = self.names(),
            ) from e
    
    def as_list(self) -> list[langchain_core.tools.BaseTool]:
        '''Get a list of tools.'''
        return list(self.tools.values())

    def render(self) -> str:
        '''Gets description of toolset using the render method.'''
        return langchain_core.tools.render.render_text_description(self.as_list())
    
    def names(self) -> list[str]:
        return list(self.tools.keys())
    