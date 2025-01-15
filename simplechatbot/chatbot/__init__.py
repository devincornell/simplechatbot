# whichever chatbot is here will be imported into 
#   the simplechatbot primary namespace
#   so you can access it like this:
#   import simplechatbot
#   simplechatbot.Chatbot
#from .v4 import *


# when there are more versions, we can do this:
from .chatbot import ChatBot
from .toolset import ToolSet, ToolCallResult
from .message_history import MessageHistory
from .keychain import APIKeyChain
from .errors import UknownToolError, ToolRaisedExceptionError, ToolWasNotExecutedError
from .chatresult import ChatResult, ChatStream
# import old stuff into separate namespace
#from . import v4

from .structbot import StructBot
