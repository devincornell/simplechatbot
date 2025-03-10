import sys
sys.path.append('../src/')

import simplechatbot

#from langchain_openai import ChatOpenAI
from simplechatbot.openai_agent import OpenAIAgent
from simplechatbot.ollama_agent import OllamaChatBot


def new_ollama_agent(**kwargs) -> OllamaChatBot:
    return OllamaChatBot.new(
        model_name = 'dolphin-mixtral:8x7b',
        **kwargs,
    )

def new_openai_agent(**kwargs) -> OpenAIAgent:
    keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')
    return OpenAIAgent.new(
        model_name = 'gpt-4o-mini',
        api_key=keychain['openai'],
        **kwargs
    )

def new_dm() -> simplechatbot.ChatBot:
    return new_ollama_agent(
        system_prompt=(
            'You are a dungeon master. You are running a game of Dungeons and Dragons.'
            'Your players are about to enter a dark cave. You must describe the cave to them.'
        )
    )

def stream_msg(agent: simplechatbot.ChatBot, msg: str) -> simplechatbot.ChatResult:
    stream = agent.stream(msg, add_to_history=True)
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    return stream.collect()


def main():

    # optional: use this to grab keys from a json file rather than setting system variables
    
    #agent1, agent2 = new_dm(keychain), new_philosophy_agent(keychain)

    discussion_topic = 'Is it better for humans to grow our population or not?'


    print(f'AGENT: {agent1.history.system_prompt}')

    print(f'DISCUSSION TOPIC: {discussion_topic}\n\n\n')

    content = discussion_topic
    for i in range(int(1e3)):
        print('='*40, f'Agent 1 Response {i+1}', '='*40)
        content = agent1.stream(content).content
        print('\n\n\n')

        print('='*40, f'Agent 2 Response {i+1}', '='*40)
        content = stream_msg(agent2, content).content
        print('\n\n\n')

if __name__ == '__main__':
    main()


