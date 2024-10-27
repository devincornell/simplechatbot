# whichever chatbot is here will be imported into 
#   the simplechatbot primary namespace
#   so you can access it like this:
#   import simplechatbot
#   simplechatbot.Chatbot
#from .v4 import *


# when there are more versions, we can do this:
from .chatbot import ChatBot
from .toolset import ToolSet, UknownToolError, ToolRaisedExceptionError, ToolCallResult
from .message_history import MessageHistory
from .keychain import APIKeyChain

# import old stuff into separate namespace
from . import v4

