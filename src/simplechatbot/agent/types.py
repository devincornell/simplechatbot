import typing
ToolCallID = str
ToolName = str

class UnspecifiedType:
    pass

UNSPECIFIED = UnspecifiedType()

AgentID = typing.Union[int, str]
