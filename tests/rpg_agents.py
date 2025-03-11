from __future__ import annotations
import typing
import dataclasses
import pydantic
import pathlib

import sys
sys.path.append('../src/')
import simplechatbot

################################################ Agent base classes ################################################

class InteractingAgent:
    '''Base class for any agent that interacts with other agents.'''
    agent: simplechatbot.Agent

    def listen(self, message: str) -> None:
        '''Listen to a message from another agent.'''
        return self.agent.history.add_human_message(message)
    
    def speak(self, message: str) -> simplechatbot.StreamResult:
        '''Speak to another agent.'''
        return self.agent.stream(new_message=message, add_to_history=True)


################################################ DM Agents ################################################

@dataclasses.dataclass
class SettingMakerAgent:
    agent: simplechatbot.Agent
    system_prompt = (
        'You are a dungeon master for a role playing game and you need to write an introduction '
        'for a new campaign. The campaign is set in a world where magic is real and the players '
        'are about to embark on a quest , but you need to decide on the setting and aspects of the '
        'world that will be important for the players to know. The user will give you a description '
        'of a world, and you have to come up with a situation that players must overcome with their '
        'skills and abilities. The description should be detailed and include an element of tension '
        'or conflict that the players will have to resolve. You should ONLY share the setting  and '
        'story description with the user - do not describe the player characters or their actions, '
        'only the background characters.'
    )

    @classmethod
    def write_intro(cls, base_agent: simplechatbot.Agent, description: str) -> str:
        '''Write a new random character description.'''
        agent = base_agent.new_agent_from_model(
            system_prompt = cls.system_prompt,
        )
        return agent.chat(
            new_message=f'Write a background introduction for a new campaign loosely based on this idea: {description}',
            add_to_history=False,
        ).content

@dataclasses.dataclass
class DMAgent(InteractingAgent):
    agent: simplechatbot.Agent
    system_prompt = (
        'You are a dungeon master running a role-playing game with one or more players.'
        'You spent time writing an introduction to a setting/scene, and are excited to get '
        'the players started on their adventure. The game goes like this: each turn, a player '
        'will describe what their character does, and you will respond with the outcome of their '
        'actions. You can also introduce new characters, settings, and plot twists to keep the '
        'game interesting. The goal is to create an engaging and immersive experience for the players.'
        'The game progresses as each player takes their turn, and it is up to you to decide '
        'the consequences of those actions. '
        'This is the introduction to the scene and setting of the game:\n{introduction}'
    )

    @classmethod
    def from_random(cls, base_agent: simplechatbot.Agent, introduction: str) -> typing.Self:
        '''Create a random DM agent.'''
        intro_text = SettingMakerAgent.write_intro(base_agent, introduction)
        return cls.from_setting(base_agent, intro_text)

    @classmethod
    def from_setting(cls, base_agent: simplechatbot.Agent, introduction: str) -> typing.Self:
        return cls(
            agent=base_agent.new_agent_from_model(
                system_prompt = cls.system_prompt.format(introduction=introduction)
            ),
        )
    
    def explain_introduction(self) -> simplechatbot.StreamResult:
        return self.agent.stream(
            new_message=(
                'Give an introduction to the game!'
            ),
            add_to_history=False,
        )
    
    def listen_to_character(self, char: CharacterAgent, msg: str) -> None:
        '''Listen to a message from a character agent.'''
        return self.agent.history.add_human_message(f'{char.name}: {msg}')

    def write_history_file(self, path: pathlib.Path) -> None:
        with path.joinpath('dm.txt').open('w') as f:
            f.write(self.agent.history.as_string())

################################################ Character Agents ################################################
class CharacterAttributes:
    '''Character attributes for a role-playing game character.'''
    name: str = pydantic.Field(description='The name of the character.')
    description: str = pydantic.Field(description='The description of the character.')

    @property
    def id(self) -> str:
        return '_'.join(self.name.lower().split())

@dataclasses.dataclass
class CharacterDescriptionAgent:
    agent: simplechatbot.Agent
    character_description: str
    system_prompt = (
        'You are designed to creat a personality and background description for a character in a '
        'a role-playing game that is being run by a dungeon master. You may create any character at '
        'all, but you should try to make them interesting and engaging. The character should have a '
        'clear personality, background, and motivations that will influence their actions in the game. '
        'You should also consider how the character will interact with other characters and the world '
        'around them when creating the description. Make the character one that would appear in a '
        'fantasy or science fiction setting, but the character can be a unique blend of elements from '
        'different genres.'
    )

    @classmethod
    def write_character_description(cls, base_agent: simplechatbot.Agent) -> CharacterAttributes:
        '''Write a new random character description.'''
        agent = base_agent.new_agent_from_model(
            system_prompt = cls.system_prompt,
        )
        return agent.chat_structured(
            new_message=None,
            output_structure=CharacterAttributes,
            add_to_history=False,
        ).data

@dataclasses.dataclass
class CharacterAgent(InteractingAgent):
    agent: simplechatbot.Agent
    attributes: CharacterAttributes

    system_prompt = (
        'You are a character in a role-playing game that is being run by a dungeon master. '
        'The dungeon master will put you in a place and setting, possibly with other characters, '
        'and you will have to decide what your character does in response to the situation. '
        'You can interact with other characters, explore the setting, and take actions that '
        'influence the outcome of the game. The dungeon master will describe the world around you, '
        'and you will have to use your imagination to decide how your character reacts. '
        'The goal is to create an engaging and immersive experience for everyone involved. \n'
        'You are to take actions according to your character\'s personality and abilities, '
        'and the dungeon master will respond with the outcome of those actions. \n\n'
        'Your character name: {name}\n'
        'Your character personality: {description}'
    )
    @classmethod
    def from_random(cls, base_agent: simplechatbot.Agent) -> typing.Self:
        '''Create a random character agent.'''
        attributes = CharacterDescriptionAgent.write_character_description(base_agent)
        return cls.from_attributes(base_agent, attributes)

    @classmethod
    def from_attributes(cls, base_agent: simplechatbot.Agent, attributes: CharacterAttributes) -> typing.Self:
        return cls(
            agent=base_agent.new_agent_from_model(
                system_prompt = cls.system_prompt.format(
                    name = attributes.name,
                    description = attributes.description,
                )
            ),
            attributes=attributes,
        )
    
    def give_introduction(self) -> simplechatbot.StreamResult:
        return self.agent.stream(
            new_message=(
                'Introduce yourself to the other characters!',
            ),
            add_to_history=True,
        )
    
    @property
    def id(self) -> str:
        return self.attributes.id
    
    @property
    def name(self) -> str:
        return self.attributes.name
    
    @property
    def description(self) -> str:
        return self.attributes.description


class CharacterAgents(typing.List[CharacterAgent]):
    @classmethod
    def create_random_characters(cls, base_agent: simplechatbot.Agent, n: int) -> typing.Self:
        '''Create a list of n character agents with llm-generated character names and descriptions.'''
        return cls(CharacterAgent.from_random(base_agent) for _ in range(n))

    def listen_to_dm(self, message: str) -> typing.List[simplechatbot.StreamResult]:
        '''Listen to a message from another agent.'''
        return [agent.listen(f'From the Dungeon Master: {message}') for agent in self]

    def listen_to_others(self, speaker: CharacterAgent, message: str) -> typing.List[simplechatbot.StreamResult]:
        '''Listen to a message from another agent.'''
        return [agent.listen(f'{speaker.name}: {message}') for agent in self if agent.id != speaker.id]

    def write_history_files(self, path: pathlib.Path) -> None:
        for c in self:
            with path.joinpath(f'{c.id}.txt').open('w') as f:
                f.write(c.agent.history.as_string())

