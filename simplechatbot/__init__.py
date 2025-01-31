
# so this places everything from the tools module into the tools namespace. Access it like this:
# import simplechatbot
# simplechatbot.tools.WhateverTool
from . import tools

from . import devin
from . import andrew

# I had to move this to the package root for some imports to work.
from .chatbot import *

from .promptmanager import PromptManager, PromptNotFound, TemplateVariableMismatch
