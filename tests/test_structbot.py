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


    storybot = simplechatbot.StructBot.from_model(
        model = ChatOpenAI(
            name = 'gpt-4o-mini', 
            api_key=simplechatbot.APIKeyChain.from_json_file('../keys.json')['openai'],
        ),
        system_prompt=system_prompt,
        output_structure = Story,
    )

    story = storybot.invoke(
        new_message='A dog really wants to befriend a chicken but it is somewhat scared of the chicken.',
    )
    
    print(f'{"="*10}\n{story.title}\n\n{story.body}\n{"="*10}')


    class Story2(pydantic.BaseModel):
        title: str = pydantic.Field(description="The title of the story.")
        summary: str = pydantic.Field(description="A brief summary of the story.")
        body: str = pydantic.Field(description="The body of the story.")


    storybot2 = storybot.clone(output_structure=Story2)
    story = storybot2.invoke(
        new_message='Close friend is leaving to study overseas and you are very sad :(',
    )
    print(f'{"="*10}\nTitle: {story.title}\n\nSummary: {story.summary}\n\n{story.body}\n{"="*10}')

    # testing empty
    assert(len(storybot2.empty(keep_system_prompt=False).history) == 0)
    assert(len(storybot2.empty(keep_system_prompt=True).history) == 1)

if __name__ == '__main__':
    test_structbot()



