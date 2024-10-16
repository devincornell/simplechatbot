

from __future__ import annotations

import typing
import dataclasses

import langchain_core.tools
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import pydantic
import json

# pip install -qU langchain-google-community\[gmail\]
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)


import sys
sys.path.append('..')
import simplechatbot
from simplechatbot.devin.openai import OpenAIChatBot

def get_gmail_tools() -> list[str]:
    '''Get the tools available for the model.'''


    # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
    # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)

    return toolkit.get_tools()
    

    

if __name__ == '__main__':

    # keychain is now just a dict subclass
    keychain = simplechatbot.devin.APIKeyChain.from_json_file('../keys.json')

    system_prompt = '''
    You are an email assistant designed to help the user send and search through emails.
    To every email the user wants to send, add "sent from my email assistant" at the end.
    '''
    chatbot = OpenAIChatBot.new(
        model_name = 'gpt-4o-mini', 
        api_key=keychain['openai'],
        system_prompt=system_prompt,
        tools=get_gmail_tools(), # just includes the rag tool from above
    )
    

    print('=============== Starting Chat ===================\n')
    
    chatbot.ui.start_interactive(stream=True, show_intro=True, show_tools=True)

    print('=============== Chat Ended ===================')

