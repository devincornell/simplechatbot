

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
import simplechatbot

    
if __name__ == '__main__':

    # keychain is now just a dict subclass
    keychain = simplechatbot.devin.APIKeyChain.from_json_file('keys.json')

    system_prompt = '''
    You are designed to answer any question the user has, to the best of your ability.
    At some point, the user may ask you to save or retrieve text to or from a workspace.
    You should ONLY do this if the user explicitly asks you to do so.
    '''
    chatbot = simplechatbot.devin.ChatBot.from_openai(
        model_name = 'gpt-4o-mini', 
        api_key=keychain['openai'],
        system_prompt=system_prompt,
        toolkits = [simplechatbot.tools.WorkspacesToolkit()], # just includes the rag tool from above
    )
    

    print('=============== Starting Chat ===================\n')
    
    chatbot.ui.start_interactive(stream=True, show_intro=True, show_tools=True)

    print('=============== Chat Ended ===================')

