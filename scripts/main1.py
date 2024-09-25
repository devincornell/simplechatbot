
import typing
import dataclasses

import langchain_core.tools
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import pydantic

#pip install duckduckgo-search
from langchain_community.tools import DuckDuckGoSearchResults


import sys
sys.path.append('..')

import simplechatbot.v4 as simplechatbot


def get_tools() -> list[langchain_core.tools.BaseTool]:
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

    return [
        DuckDuckGoSearchResults(
            keys_to_include=['snippet', 'title'], 
            results_separator='\n\n',
            num_results = 10,
        ),
        multiply,
        add,
    ]


if __name__ == '__main__':

    system_prompt = '''
    You are designed to answer any question the user has by browsing the web with Duck Duck Go Search.
    Even if the user did not ask for it, perform a search to answer every question.
    '''

    if True:
        chatbot = simplechatbot.ChatBot.from_openai(
            model_name = 'gpt-4o-mini', 
            api_key=simplechatbot.APIKeyChain.from_json_file('keys.json')['openai'],
            system_prompt=system_prompt,
            tools=get_tools(),
        )
    else:
        chatbot = simplechatbot.ChatBot.from_ollama(
            model_name = 'llama3.1', 
            system_prompt=system_prompt,
            tools=get_tools(),
        )
    
    if True:
        try:
            system_prompt = chatbot.history.first_system.content
            print('=============== System Message for this Chat ===================')
            print(system_prompt, '\n')
        except ValueError as e:
            pass
        try:
            tools = chatbot.toolset.render()
            print('\n=============== Tools for this Chat ===================')
            print(tools, '\n')
        except AttributeError:
            pass

    print('=============== Starting Chat ===================\n')
    
    chatbot.ui.start_interactive(stream=False, tool_verbose_callback=print)
    
    print('=============== Chat Ended ===================')

