

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


#import importlib
#import simplechatbot.v4
#importlib.reload(simplechatbot.v4)
#import simplechatbot.v4 as simplechatbot
import simplechatbot

from simplechatbot.tools.rag.rag import RAG

    
if __name__ == '__main__':

    # keychain is now just a dict subclass
    keychain = simplechatbot.devin.APIKeyChain.from_json_file('keys.json')


    # I put all the RAG state in this object
    rag = RAG.from_web_pages(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        nvidia_api_key=keychain['nvidia'],
    )

    # I added this to convert it to a tool
    rag_tool = rag.as_tool(
        name = 'Lilian Weng Blog Retriever',
        description = 'Retrieves relevant text from Lilian Wengs blog to answer any questions the tool has..'
    )

    system_prompt = '''
    You are designed to answer any question the user has.
    For every question, you should retrieve test from Lilian Wengs blog.
    '''
    if "openai" in keychain:
        chatbot = simplechatbot.devin.ChatBot.from_openai(
            model_name = 'gpt-4o-mini', 
            api_key=keychain['openai'],
            system_prompt=system_prompt,
            tools = [rag_tool], # just includes the rag tool from above
        )
    else:
        chatbot = simplechatbot.devin.ChatBot.from_ollama(
            model_name = 'llama3.1', 
            system_prompt=system_prompt,
            tools = [rag_tool],
        )
    

    print('=============== Starting Chat ===================\n')
    
    chatbot.ui.start_interactive(stream=True, show_intro=True, show_tools=True)

    print('=============== Chat Ended ===================')

