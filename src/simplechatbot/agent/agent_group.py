from __future__ import annotations

import typing
import dataclasses
import copy

import pydantic

from langchain_core.messages import (
    AIMessageChunk, 
    AIMessage, 
    BaseMessage, 
    HumanMessage,
)
from langchain_core.tools import (
    Tool,
)

from .message_history import MessageHistory
from .toolset import ToolSet, ToolCallResult, ToolLookup

from .agent import Agent
from .types import AgentID


def my_tool_function(input: str) -> str:
    # Your tool's logic here
    return f"Processed input: {input}"

my_tool = Tool(
    name="MyDynamicTool",
    func=my_tool_function,
    description="A tool that processes input dynamically at runtime.",
)



@dataclasses.dataclass(repr=False)
class AgentGroup:
    agents: dict[AgentID, Agent]

    @classmethod
    def from_agent_dict(cls, agents: dict[AgentID, Agent]) -> AgentGroup:
        '''Create an agent group from a dictionary of agents.'''
        return cls(
            agents=agents,
        )
    
    def get_interaction_tools(self) -> dict[int|str, ToolSet]:
        '''Get the interaction tools for each agent.'''
        for aid in self.agents:
            print('hello world')
        return {agent_id: agent.toolset.get_interaction_tools() for agent_id, agent in self.agents.items()}


    @classmethod
    def get_interaction_tool(cls, omit_agent_id: AgentID) -> ToolSet:
        '''Get the interaction tools for each agent.'''
        for aid in self.agents:
            if aid != omit_agent_id:
                return self.agents[aid].toolset.get_interaction_tools()
        return {agent_id: agent.toolset.get_interaction_tools() for agent_id, agent in self.agents.items() if agent_id != omit_agent_id}

