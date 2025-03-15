import typing
import dataclasses
import pathlib

import sys
sys.path.append('../src/')
import simplechatbot
from simplechatbot.openai_agent import OpenAIAgent
from simplechatbot.ollama_agent import OllamaAgent

import rpg_agents


def new_ollama_agent(**kwargs) -> OllamaAgent:
    return OllamaAgent.new(
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




def main():
    outpath = pathlib.Path('test1/')
    outpath.mkdir(exist_ok=True)

    if use_local := True:
        base_agent = OllamaAgent.new(model_name = 'dolphin-mixtral:8x7b')
    else:
        keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')
        base_agent = OpenAIAgent.new(model_name = 'gpt-4o-mini', api_key=keychain['openai'])
    
    #setting_maker = rpg_agents.SettingMakerAgent.from_base_agent(base_agent)
    #story_intro = setting_maker.make_game_introduction(
    #    'The players are in a dark forest with a full moon shining through the trees. They hear a rustling in the bushes and see a pair of glowing eyes staring at them.'
    #).print_and_collect()
    topic = 'The setting is a dark forest with a full moon shining through the trees. The players hear a rustling in the bushes and see a pair of glowing eyes staring at them.'
    dm_agent = rpg_agents.DMAgent.from_random(base_agent, topic)
    characters = rpg_agents.CharacterAgents.create_random_characters(base_agent, 2)

    # dm introduces the game
    intro = dm_agent.explain_introduction().print_and_collect()
    characters.listen_to_dm(intro.content)

    # each character introduces thsemselsves
    for char in characters:
        intro = char.agent.chat('Introduce yourself!').content
        msg = f'My name is {char.name} and this is my introduction:\n{msg}'
        dm_agent.listen_to_character(char, msg)
        

        
    dm_agent.explain_introduction().print_and_collect()

    return
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


