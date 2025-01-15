from __future__ import annotations
import typing
import getpass
import os
import tqdm

import langchain_core.tools

from langchain_community.agent_toolkits import FileManagementToolkit
import tempfile #python standard library

from langchain_openai import ChatOpenAI
import pydantic

import sys
sys.path.append('..')
import simplechatbot


def test_structbot():


    system_prompt = '''
    The user will give a brief description of a story and you are to create a short story with a title and body.
    Make the story creative and original.
    '''

    class Story(pydantic.BaseModel):
        title: str = pydantic.Field(description="The title of the story.")
        body: str = pydantic.Field(description="The body of the story.")


    structbot = simplechatbot.StructBot.from_model(
        model = ChatOpenAI(
            name = 'gpt-4o-mini', 
            api_key=simplechatbot.APIKeyChain.from_json_file('../keys.json')['openai'],
        ),
        system_prompt=system_prompt,
        output_structure = Story,
    )

    story = structbot.invoke(
        new_message='A dog really wants to befriend a chicken but it is somewhat scared of the chicken.',
    )
    
    
    print(f'{story.title}\n\n{story.body}')







if __name__ == '__main__':
    test_structbot()



