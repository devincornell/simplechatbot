import typing
import dataclasses
import json

import pydantic
from langchain_core.tools import BaseTool, BaseToolkit
import langchain_core.tools

# data stored behind the scenes at the module level

@dataclasses.dataclass
class WorkspacesToolkit:
    '''Represents a workspace where the user can insert and remove items.'''
    workspace_data: dict[int,tuple[str,str]] = dataclasses.field(default_factory=dict)
    ctr: int = 0

    def get_tools(self) -> list[BaseTool]:
        return [
            self.tool_insert_item(),
        ]
    
    def view_workspaces(self) -> BaseTool:
        '''Input to the function to list all availble workspace ids and summaries.'''

        @langchain_core.tools.tool("view_workspaces")
        def view_workspaces() -> str:
            """View all available workspace ids and their summaries as a json string."""
            #self.workspace_data[self.ctr] = ((title, description))
            #self.ctr += 1
            #return f"Workspace saved with id {self.ctr}. Access it using that id."
            data = list()
            for idx, (summary, full_text) in self.workspace_data.items():
                data.append({'id': idx, 'summary': summary})
            return json.dumps(data)
        
        return view_workspaces

    def tool_insert_item(self) -> BaseTool:
        '''Insert an item into the workspace.'''

        class SaveWorkspaceInput(pydantic.BaseModel):
            """Inputs to the function to save text to the workspace."""
            summary: str = pydantic.Field(
                description="Brief summary of the workspace data being saved."
            )
            description: str = pydantic.Field(
                description="Full text of workspace data to save."
            )

        @langchain_core.tools.tool("save_workspace", args_schema=SaveWorkspaceInput)
        def save_workspace(summary: str, description: str) -> str:
            """Save a workspace. The workspace contains summary and full text information."""
            self.workspace_data[self.ctr] = ((summary, description))
            self.ctr += 1
            return f"Workspace saved with id {self.ctr} with summary: '{summary}'."
        
        return save_workspace
        
    def tool_retrieve_workspace(self) -> BaseTool:
        '''Insert an item into the workspace.'''

        class RetrieveWorkspaceInput(pydantic.BaseModel):
            """Inputs to the function to retrieve notes."""
            id: int = pydantic.Field(
                description="ID of the workspace to retrieve."
            )

        @langchain_core.tools.tool("retrieve_workspace", args_schema=RetrieveWorkspaceInput)
        def retrieve_workspace(id: int) -> str:
            """Retrieve a particular workspace by its id."""
            return self.workspace_data[id]
                
        return retrieve_workspace

    
    