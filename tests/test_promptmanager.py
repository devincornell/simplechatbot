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
#from simplechatbot.openai import OpenAIChatBot
from simplechatbot.promptmanager import jinja_get_variables

import pytest

def test_manager():
    pman = simplechatbot.PromptManager('test_prompts')
    prompt = pman.get_prompt('test1')
    print(prompt)
    assert('ahsdkljfhhjklashd' in prompt)

    prompt2 = pman.get_prompt(
        path='test2.txt',
        template_vars={'answer': 'Alice'},
    )
    assert('Alice' in prompt2)
    print(prompt2)
    print(jinja_get_variables(prompt2))

    with pytest.raises(simplechatbot.TemplateVariableMismatch) as exc_info:
        prompt3 = pman.get_prompt(
            path='test2',
            #template_vars={'answer': 'Alice'},
        )

    with pytest.raises(simplechatbot.TemplateVariableMismatch) as exc_info:
        prompt3 = pman.get_prompt(
            path='test2',
            template_vars={'answer': 'Alice', 'whateva': 'whatever'},
        )

    with pytest.raises(simplechatbot.TemplateVariableMismatch) as exc_info:
        prompt3 = pman.get_prompt(
            path='test2',
            template_vars={'whateva': 'Alice'},
        )


if __name__ == '__main__':
    test_manager()

