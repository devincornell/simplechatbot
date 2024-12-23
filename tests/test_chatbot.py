from __future__ import annotations
import typing
import getpass
import os

import langchain_core.tools

from langchain_community.tools import DuckDuckGoSearchResults

import sys
sys.path.append('..')
import simplechatbot
from simplechatbot.devin.openai import OpenAIChatBot

TEST_NUMBER = '324'

def get_tools():
    @langchain_core.tools.tool
    def send_message(text: str, username: str) -> str:
        '''Use this tool when you want to send a message to a user.'''
        return f'Message sent to {username}: id={TEST_NUMBER}'
    
    @langchain_core.tools.tool
    def check_new_messages(text: str, username: str) -> str:
        '''Check messages.'''
        return f'No new messages.'
    

    return [
        send_message, 
        check_new_messages, 
    ]

def get_toolkits():
    return []


def test_tools():
    keychain = simplechatbot.devin.APIKeyChain.from_json_file('../keys.json')

    system_prompt = '''
    You are designed to answer any question the user has and send/check messages if needed.
    When the user requests you to send a message to a user, 
     you should send the message and tell the user the resulting id.
    When the user requests information that may be out-of-date for your tools,
        browse the web to find the most recent information.
    '''
    chatbot = OpenAIChatBot.new(
        model_name = 'gpt-4o', 
        api_key=keychain['openai'],
        system_prompt=system_prompt,
        tools = get_tools(),
    )
    print(chatbot)


    r = chatbot.invoke('How are you today?')
    assert(len(r.content))
    assert(len(r.tool_calls) == 0)

    r = chatbot.invoke('Browse the web and tell me the weather today in Rome.')
    assert(len(r.content))
    assert(len(r.tool_calls) == 0)

    r = chatbot.invoke(
        'Browse the web and tell me the weather today in Rome.', 
        tools=[DuckDuckGoSearchResults()],
    )
    assert(len(r.tool_calls) == 1)
    assert('duckduckgo_results_json' == r.tool_calls[0].name)

    r = chatbot.invoke(f'Send the following message to @devin: "Hello, how are you?"')
    assert('send_message' == r.tool_calls[0].name)
    
    #r = chatbot.chat('', add_to_history=False)
    return
    
    reply_stream = chatbot.chat_stream(f'Send the following message to @devin: "Hello, how are you?"')
    for r in reply_stream:
        print(r.content, end='', flush=True)
        len(r.content)
    print()
    print(reply_stream.tool_calls)


    
    r = chatbot.invoke('Can you do a web search to find the latest weather in Leon, IA?')
    print(r.content)
    assert(len(r.content))
    print(r.tool_calls)
    assert(len(r.tool_calls) == 1)
    

    assert(len(reply_stream.tool_calls) == 1)
    results = reply_stream.call_tools()
    print(results['message_tool'].return_value)

    for r in chatbot.chat_stream(new_message=None):
        print(r.content, end='', flush=True)
    #chatbot.ui.start_interactive(stream=True, show_tools=False)



if __name__ == '__main__':
    test_tools()