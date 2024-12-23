from __future__ import annotations

import typing
import langchain_core.tools

from .types import ToolCallID
from .util import format_tool_text

if typing.TYPE_CHECKING:
    from .toolset import ToolCallInfo

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
    tool_info: ToolCallInfo
    e: Exception

    @classmethod
    def from_exception(cls, tool_info: ToolCallInfo, e: Exception) -> typing.Self:
        o = cls(f'{tool_info.tool_info_str()} raised an exception: {e}')
        o.tool_info = tool_info
        o.e = e
        return o
