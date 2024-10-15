from __future__ import annotations
import typing
import getpass
import os

import langchain_core.tools

import sys
sys.path.append('..')
import simplechatbot

def get_tools():
    @langchain_core.tools.tool
    def message_tool(text: str) -> str:
        '''Send a message to the system.'''
        return 'Message sent: ' + text
    return [message_tool]


if __name__ == '__main__':
    keychain = simplechatbot.devin.APIKeyChain.from_json_file('../scripts/keys.json')

    system_prompt = '''
    You are designed to answer any question the user has.
    '''
    chatbot = simplechatbot.devin.ChatBot.from_openai(
        model_name = 'gpt-4o-mini', 
        api_key=keychain['openai'],
        system_prompt=system_prompt,
        tools = get_tools(),
    )
    print(chatbot)

    chatbot.ui.start_interactive(stream=True, show_tools=False)


