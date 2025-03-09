import sys
sys.path.append('../src/')

import simplechatbot

#from langchain_openai import ChatOpenAI
#from simplechatbot.openai_chatbot import OpenAIChatBot
from simplechatbot.ollama_chatbot import OllamaChatBot


def new_random_agent(keychain: simplechatbot.APIKeyChain) -> simplechatbot.ChatBot:
    return OllamaChatBot.new(
        model_name = 'dolphin-mixtral:8x7b',
        #model_name='gpt-4o-mini', 
        #api_key=keychain['openai'],
        system_prompt=(
            'Your name is Devin.'
        )
    )

def stream_msg(agent: simplechatbot.ChatBot, msg: str) -> simplechatbot.ChatResult:
    stream = agent.chat_stream(msg, add_to_history=True)
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    return stream.result()


def main():

    # optional: use this to grab keys from a json file rather than setting system variables
    keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')
    agent = new_random_agent

    discussion_topic = 'Is it better for humans to grow our population or not?'


    print(f'AGENT: {agent1.history.system_prompt}')

    print(f'DISCUSSION TOPIC: {discussion_topic}\n\n\n')

    content = discussion_topic
    for i in range(int(1e3)):
        print('='*40, f'Agent 1 Response {i+1}', '='*40)
        content = stream_msg(agent1, content).content
        print('\n\n\n')

        print('='*40, f'Agent 2 Response {i+1}', '='*40)
        content = stream_msg(agent2, content).content
        print('\n\n\n')

if __name__ == '__main__':
    main()


