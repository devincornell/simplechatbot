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
    id: ToolCallID
    tool: langchain_core.tools.BaseTool
    value: typing.Any
    tool_info: dict[str, str|dict]

    @property
    def tool_name(self) -> str:
        return self.tool.name

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
        verbose_callback: typing.Optional[typing.Callable[[str],None]] = lambda x: None
    ) -> ToolCallResult:
        '''Extracts tool information and executes tool.'''
        name = tool_info['name']
        tool = self[name] # raises UnidentifiedToolError if tool isn't found

        # actually execute tool call!
        # NOTE: This happens outside the LLM, so can do anything with Python
        args = tool_info['args']

        try:
            value = tool.invoke(args)
        except Exception as e:
            raise ToolRaisedExceptionError.from_exception(tool_info, tool, e) from e
        
        # print tool output?
        if verbose_callback is not None:
            #verbose_callback(f'function: {format_tool_text(tool_info)} -> {value}')
            verbose_callback(f'tool call: {format_tool_text(tool_info)}')
        
        return ToolCallResult(
            id=tool_info['id'],
            value = value, 
            tool = tool,
            tool_info = tool_info, 
        )

    def __getitem__(self, name: str) -> langchain_core.tools.BaseTool:
        '''Get a tool by name.'''
        try:
            return self.tools[name]
        except KeyError as e:
            raise UknownToolError.from_tooL_name(
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
    


