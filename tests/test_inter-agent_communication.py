import sys
sys.path.append('../src/')

import simplechatbot

#from langchain_openai import ChatOpenAI
#from simplechatbot.openai_chatbot import OpenAIChatBot
from simplechatbot.ollama_agent import OllamaAgent


def new_philosophy_agent(keychain: simplechatbot.APIKeyChain) -> simplechatbot.ChatBot:
    return OllamaAgent.new(
        model_name = 'dolphin-mixtral:8x7b',
        #model_name='gpt-4o-mini', 
        #api_key=keychain['openai'],
        system_prompt=(
            'Your job is to write amazing science fiction stories. The user will first give you '
            'ideas and then iterate to come up with new stories, which you can take and iterate on again. '
            'Through repeated back-and-forth iteration you will be able to come up with amazing stories.'
            'Feel free to pull ideas from every possible science fiction story you can think of. '
            'The best story ideas will include interesting commentary on our present society and '
            'include critiques of the contradictions that are prevalent in our society. These critiques '
            'should be expressed inherently as part of the narrative rather than mentioned explicitly. '
            'Great stories are also very long, so be sure to produce really long stories.'
        )
    )

def stream_msg(agent: simplechatbot.ChatBot, msg: str) -> simplechatbot.ChatResult:
    stream = agent.stream(msg, add_to_history=True)
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    return stream.collect()


def main():

    # optional: use this to grab keys from a json file rather than setting system variables
    keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')
    agent1, agent2 = new_philosophy_agent(keychain), new_philosophy_agent(keychain)

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


