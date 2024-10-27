from __future__ import annotations
import typing
import getpass
import os

import langchain_core.tools

import sys
sys.path.append('..')
import simplechatbot
from simplechatbot.devin.openai import OpenAIChatBot

def get_tools():
    @langchain_core.tools.tool
    def message_tool(text: str) -> str:
        '''Send a message to the system.'''
        return 'Message sent: ' + text
    return [message_tool]


if __name__ == '__main__':
    keychain = simplechatbot.util.APIKeyChain.from_json_file('../keys.json')

    system_prompt = '''
    You are designed to answer any question the user has.
    '''
    chatbot = OpenAIChatBot.new(
        model_name = 'gpt-4o-mini', 
        api_key=keychain['openai'],
        system_prompt=system_prompt,
        tools = get_tools(),
    )
    print(chatbot)

    reply_stream = chatbot.chat_stream(f'Send the following message: "Hello, how are you?"')
    for r in reply_stream:
        print(r.content, end='', flush=True)

    assert(len(reply_stream.tool_calls) == 1)
    results = reply_stream.call_tools()
    print(results['message_tool'].return_value)

    for r in chatbot.chat_stream(new_message=None):
        print(r.content, end='', flush=True)
    #chatbot.ui.start_interactive(stream=True, show_tools=False)


