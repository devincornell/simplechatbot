from __future__ import annotations
import typing
import getpass
import os
import tqdm

import langchain_core.tools

from langchain_community.agent_toolkits import FileManagementToolkit
import tempfile #python standard library


import sys
sys.path.append('..')
import simplechatbot
from simplechatbot.openai import OpenAIChatBot



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

def get_toolkits(wd: str):
    return [
        FileManagementToolkit(
            root_dir=str(wd)
        ),
    ]


def test_tools():
    keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')

    system_prompt = '''
    You are designed to answer any question the user has and send/check messages if needed.
    When the user requests you to send a message to a user, 
     you should send the message and tell the user the resulting id.
    When the user requests information that may be out-of-date for your tools,
        browse the web to find the most recent information.
    '''

    with tempfile.TemporaryDirectory() as wd:
        fname = 'myname.txt'

        with open(wd + f'/{fname}', 'w') as f:
            f.write('My name is Devin!')

        chatbot = OpenAIChatBot.new(
            model_name = 'gpt-4o-mini', 
            api_key=keychain['openai'],
            system_prompt=system_prompt,
            tools = get_tools(),
            #toolkits=get_toolkits(),
        )
        print(chatbot)

        with tqdm.tqdm() as pbar:
            ########################## invoke/stream tool tests ##########################
            r = chatbot.invoke('How are you today?')
            assert(len(r.content))
            assert(len(r.tool_calls) == 0)
            pbar.update()

            r = chatbot.stream('How are you today?').result()
            assert(len(r.content))
            assert(len(r.tool_calls) == 0)
            pbar.update()

            r = chatbot.invoke('List the files in the current directory.')
            assert(len(r.content))
            assert(len(r.tool_calls) == 0)
            pbar.update()

            r = chatbot.stream('List the files in the current directory.').result()
            assert(len(r.content))
            assert(len(r.tool_calls) == 0)
            pbar.update()

            r = chatbot.invoke(
                'List the files in the current directory.', 
                toolkits=get_toolkits(wd),
            )
            assert(len(r.tool_calls) == 1)
            assert('list_directory' == r.tool_calls[0].name)
            pbar.update()

            r = chatbot.stream(
                'List the files in the current directory.', 
                toolkits=get_toolkits(wd),
            ).result()
            assert(len(r.tool_calls) == 1)
            assert('list_directory' == r.tool_calls[0].name)
            pbar.update()

            r = chatbot.invoke(f'Send the following message to @devin: "Hello, how are you?"')
            assert('send_message' == r.tool_calls[0].name)
            pbar.update()
            
            r = chatbot.stream(f'Check my messages and tell me all of the new ones.').result()
            print(r.tool_calls)
            assert('check_new_messages' == r.tool_calls[0].name)
            pbar.update()


def test_tools_chat():
    keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')

    system_prompt = '''
    You are designed to answer any question the user has and send/check messages if needed.
    When the user requests you to send a message to a user, 
     you should send the message and tell the user the resulting id.
    When the user requests information that may be out-of-date for your tools,
        browse the web to find the most recent information.
    '''

    with tempfile.TemporaryDirectory() as wd:
        fname = 'myname.txt'

        with open(wd + f'/{fname}', 'w') as f:
            f.write('My name is Devin!')

        chatbot = OpenAIChatBot.new(
            model_name = 'gpt-4o-mini', 
            api_key=keychain['openai'],
            system_prompt=system_prompt,
            tools = get_tools(),
            #toolkits=get_toolkits(wd),
        )
        print(chatbot)

        with tqdm.tqdm() as pbar:
            ########################## invoke/stream tool tests ##########################
            r = chatbot.invoke('How are you today?')
            assert(len(r.content))
            assert(len(r.tool_calls) == 0)
            pbar.update()
            
            
            r = chatbot.chat('How are you today?', add_to_history=False)
            pbar.update()

            stream = chatbot.chat_stream('How are you today?', add_to_history=False)
            for r in stream:
                pass
            assert(len(stream.tool_calls) == 0)
            pbar.update()

            stream = chatbot.chat_stream('How are you today?', add_to_history=True)
            for r in stream:
                pass
            assert(len(stream.tool_calls) == 0)
            pbar.update()

            stream = chatbot.chat_stream(
                'List the files in this directory.', 
                add_to_history=True,
                toolkits=get_toolkits(wd),
            )
            for r in stream:
                pass
            assert(len(stream.tool_calls) == 1)
            assert(stream.tool_calls[0].name == 'list_directory')
            stream.execute_tools()
            pbar.update()

            stream = chatbot.chat_stream(
                'List the files in this directory.', 
                add_to_history=True,
            )
            for r in stream:
                pass
            if len(stream.tool_calls) > 0: # tool calling is not good enough for this yet
                assert(stream.tool_calls[0].name != 'list_directory')
                stream.execute_tools()
            pbar.update()


def test_chat():

    keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')

    system_prompt = '''
    You are designed to work with the filesystem of this computer.
    '''

    with tempfile.TemporaryDirectory() as wd, tqdm.tqdm() as pbar:
        fname = 'myname.txt'

        with open(wd + f'/{fname}', 'w') as f:
            f.write('My name is Devin!')

        toolkit = FileManagementToolkit(
            root_dir=str(wd)
        )

        chatbot = OpenAIChatBot.new(
            model_name = 'gpt-4o-mini', 
            api_key=keychain['openai'],
            system_prompt=system_prompt,
            tools = toolkit.get_tools(),
        )
        print(chatbot)

        stream = chatbot.chat_stream('List the files in this directory.')
        for r in stream:
            print(r.content, end='', flush=True)
        results = stream.execute_tools()
        print(results)
        text = chatbot.chat(None).content
        assert(fname in text)
        pbar.update()

        # test writing file and reading it back.
        fname = 'hello.txt'
        stream = chatbot.chat_stream(f'Write a text file called "{fname}" to disk with the content "Hello, world!"')
        for r in stream:
            print(r.content, end='', flush=True)
        print(stream.tool_calls[0].name)
        assert(stream.tool_calls[0].name == 'write_file')
        print(stream.execute_tools())
        pbar.update()

        stream = chatbot.chat_stream('List files in the directory.')
        for r in stream:
            print(r.content, end='', flush=True)
        assert(stream.tool_calls[0].name == 'list_directory')
        stream.execute_tools()
        assert(fname in chatbot.chat(None).content)
        pbar.update()
        

        r = chatbot.chat('List files in the directory.')
        assert(r.tool_calls[0].name == 'list_directory')

        # fails bc last tool call was not executed
        try:
            r = chatbot.chat('List files in the directory.')
        except simplechatbot.ToolWasNotExecutedError as e:
            #print('exception')
            pass


if __name__ == '__main__':
    test_tools()
    test_tools_chat()
    test_chat()
    