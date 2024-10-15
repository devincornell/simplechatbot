import typing
import langchain_core.tools

from .types import ToolCallID
from .util import format_tool_text

class UknownToolError(Exception):
    tool_name: str
    available_tools: str

    @classmethod
    def from_tool_name(cls, tool_name: str, available_tools: list[str]) -> typing.Self:
        o = cls(f'The tool {tool_name} was not recognized from valid tool list: {list(available_tools)}')
        o.tool_name = tool_name
        o.available_tools = available_tools
        return o

class ToolRaisedExceptionError(Exception):
    e: Exception
    tool: langchain_core.tools.BaseTool
    text: str
    tool_info: dict[str,str|dict]

    @classmethod
    def from_exception(cls, tool_info: dict, tool: langchain_core.tools.BaseTool, e: Exception) -> typing.Self:
        #o = cls(f'The "{tool.name}" tool raised an exception: {e}: {tool_info}')
        o = cls(f'{format_tool_text(tool_info)} raised an exception: {e}')
        o.e = e
        o.tool = tool
        o.text = format_tool_text(tool_info)
        o.tool_info = tool_info
        return o
