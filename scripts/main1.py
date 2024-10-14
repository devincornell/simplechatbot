from __future__ import annotations

import typing
import dataclasses

import langchain_core.tools
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import pydantic
import json

#pip install duckduckgo-search
from langchain_community.tools import DuckDuckGoSearchResults

import sys
sys.path.append('..')


import importlib
#import simplechatbot.v4
#importlib.reload(simplechatbot.v4)
#import simplechatbot.v4 as simplechatbot
import simplechatbot

# stores notes in memory
fake_note_db = list()

def get_tools(model: ChatOllama) -> list[langchain_core.tools.BaseTool]:
    '''Get the tools that this chatbot uses..'''

    # can use objects to specify structure of input
    # benefit is you can add more detailed description to help the LLM decide
    class MultiplyInputs(pydantic.BaseModel):
        """Inputs to the multiply tool."""

        a: int = pydantic.Field(
            description="First number to multiply by."
        )
        b: int = pydantic.Field(
            description="Second number to multiply by."
        )

    # input specification appears above
    @langchain_core.tools.tool("multiply", args_schema=MultiplyInputs)
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b

    @langchain_core.tools.tool
    def add(first: int, second: int) -> int:
        "Add two numbers."
        return first + second

    return [DuckDuckGoSearchResults(), multiply, add]

def note_db_tools() -> list[langchain_core.tools.BaseTool]:
    '''These tools are for saving and listing notes.'''
    class SaveNoteInput(pydantic.BaseModel):
        """Inputs to the function to save notes."""
        title: str = pydantic.Field(
            description="Brief title of the note."
        )
        description: str = pydantic.Field(
            description="Full text of note to store."
        )

    @langchain_core.tools.tool("save_note", args_schema=SaveNoteInput)
    def save_note(title: str, description: str) -> str:
        """Save a note to the database."""
        return fake_note_db.append((title, description))

    @langchain_core.tools.tool("list_available_notes")
    def list_available_notes(title: str, description: str) -> str:
        """Get a list of all available notes."""
        return '\n'.join([f'{i+1}. title="{title}"; description="{desc}"' for i, (title,desc) in enumerate(fake_note_db)])
    
    return [
        save_note,
        list_available_notes,
    ]


def builtin_tools() -> list[langchain_core.tools.BaseTool]:
    '''Get the tools that are available from langchain.'''

    return [
        DuckDuckGoSearchResults(
            keys_to_include=['snippet', 'title'], 
            results_separator='\n\n',
            num_results = 4,
        ),
    ]

# Read keys.json file
with open("keys.json") as file:
    keys = json.load(file)
    
if __name__ == '__main__':

    system_prompt = '''
    You are designed to answer any question the user has.
    '''

    if keys["openai"] != "":

        chatbot = simplechatbot.devin.ChatBot.from_openai(
            model_name = 'gpt-4o-mini', 
            api_key=simplechatbot.devin.APIKeyChain.from_json_file('keys.json')['openai'],
            system_prompt=system_prompt,
            tools = builtin_tools() + note_db_tools(),
        )

    else:

        chatbot = simplechatbot.devin.ChatBot.from_ollama(
            model_name = 'llama3.1', 
            system_prompt=system_prompt,
            tools=get_tools(model = ChatOllama),
            rag=True
        )
    

    print('=============== Starting Chat ===================\n')
    
    chatbot.ui.start_interactive(stream=True, show_intro=True, show_tools=True)

    print('=============== Chat Ended ===================')

