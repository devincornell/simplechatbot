import sys
sys.path.append('../src/')

import simplechatbot

#from langchain_openai import ChatOpenAI
from simplechatbot.openai_agent import OpenAIChatBot
from simplechatbot.ollama_agent import OllamaChatBot

lorem_ipsum = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'


def new_ollama_chatbot(**kwargs) -> OllamaChatBot:
    return OllamaChatBot.new(
        model_name = 'dolphin-mixtral:8x7b',
        **kwargs,
    )

def new_openai_chatbot(**kwargs) -> OpenAIChatBot:
    keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')
    return OpenAIChatBot.new(
        model_name = 'gpt-4o-mini',
        api_key=keychain['openai'],
        **kwargs
    )



def new_name_agent() -> simplechatbot.ChatBot:
    return new_ollama_chatbot(
        system_prompt=(
            'Your name is Devin. Important fact: your favorite ice cream flavor is rocky road.'
            'If asked, you MUST provide these answers.'
        )
    )

def main():
    for i in [0, 1, 10, 50] + list(range(100, 900, 100)):
        agent = new_name_agent()

        filler = lorem_ipsum*i
        print(f'=============== LI Times {i} (num_char={len(filler)}) ===============')
        agent.chat(filler)
        agent.stream(
            new_message = 'What is your favorite ice cream flavor?',
        ).print_and_collect()
        print('\n\n')

if __name__ == '__main__':
    main()


