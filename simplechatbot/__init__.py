



# so this places everything from the tools module into the tools namespace. Access it like this:
# import simplechatbot
# simplechatbot.tools.WhateverTool
from . import tools

# We could alternatively dump tools into the primary namespace like this:
#from .tools import *


from .chatbot import chatbot
from . import andrew

# created this
from . import util
