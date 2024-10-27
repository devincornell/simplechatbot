# `ChatBot` Objects

`ChatBot` instances maintain the three things that define a chatbot: a chat model (or runnable), chat history, and available tools / functions. In this page I will document how to use it.


```python
import sys
sys.path.append('..')

import simplechatbot
```


A `ChatBot` may be created from any [langchain chat model](https://python.langchain.com/v0.1/docs/modules/model_io/chat/) or runnable.


```python
from langchain_openai import ChatOpenAI

# optional: use this to grab keys from a json file rather than setting system variables
keychain = simplechatbot.util.APIKeyChain.from_json_file('../keys.json')

openai_model = ChatOpenAI(model='gpt-4o-mini', api_key=keychain['openai'])
chatbot = simplechatbot.chatbot.ChatBot.from_model(model=openai_model)
print(chatbot)
```

    ChatBot(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=None)


### Model-specific Chat Bots

Model-specific chatbots only differ from `ChatBot` in that they define static factory constructor methods, all named `new`. As each chat model needs to be installed separately, they must be accessed via separate imports.


```python
from simplechatbot.chatbot.ollama import OllamaChatBot

chatbot = OllamaChatBot.new(
    model_name = 'llama3.1', 
)
```

You may do the same for `ChatOpenAI` using `from_openai`.


```python
from simplechatbot.chatbot.openai import OpenAIChatBot

chatbot = OpenAIChatBot.new(
    model_name = 'gpt-4o-mini', 
    api_key=keychain['openai'],
)
```

And Mistral as well.


```python
from simplechatbot.chatbot.mistral import MistralChatBot

chatbot = MistralChatBot.new(
    model_name = 'mistral-large-latest', 
    api_key=keychain['mistral'],
)
```

### Setting System Prompt
Use the `system_prompt` parameter to initialize the chatbot with instructions.


```python
system_prompt = '''
You are a creative designer who has been tasked with creating a new slogan for a company.
The user will describe the company, and you will need to generate three slogan ideas for them.
'''
chatbot = simplechatbot.chatbot.ChatBot.from_model(
    model = openai_model,
    system_prompt=system_prompt,
)
```

### Adding Tools
The `tools` and `toolkits` parameters allow you to pass any [langchain tools](https://python.langchain.com/v0.1/docs/modules/tools/) or [toolkits](https://python.langchain.com/v0.1/docs/modules/tools/toolkits/) you want your chatbot to be able to use. You can use one of [Langchain's built-in tools](https://python.langchain.com/v0.1/docs/integrations/tools/) (such as `FileManagementToolkit`) or [define your own custom tools](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/). I will use `FileManagementToolkit` for demonstration purposes here.

**Note**: passing the tools to the chat model at this point will not bind them to the model. That step is executed before every invokation so that new tools may be added through arguments in the chat functions.


```python
from langchain_community.agent_toolkits import FileManagementToolkit
import tempfile #python standard library
working_directory = tempfile.TemporaryDirectory()
toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)
tools = toolkit.get_tools()
tools
```




    [CopyFileTool(root_dir='/tmp/tmp42vq4a65'),
     DeleteFileTool(root_dir='/tmp/tmp42vq4a65'),
     FileSearchTool(root_dir='/tmp/tmp42vq4a65'),
     MoveFileTool(root_dir='/tmp/tmp42vq4a65'),
     ReadFileTool(root_dir='/tmp/tmp42vq4a65'),
     WriteFileTool(root_dir='/tmp/tmp42vq4a65'),
     ListDirectoryTool(root_dir='/tmp/tmp42vq4a65')]



Passing the tools to `ChatBot` will create a dictionary of tools that is referenced when attempting to execute a tool call. Before every invokation, the tools will be bound to the provided model.


```python
chatbot = simplechatbot.chatbot.ChatBot.from_model(
    model = openai_model,
    tools=tools,
)
```

The tools are stored in a `ToolSet` object which is accessed via the `toolset` property. That object stores a dictionary of all tools and their parameters so they can be referenced when a tool call is requested.


```python
chatbot.toolset.names()
```




    ['copy_file',
     'file_delete',
     'file_search',
     'move_file',
     'read_file',
     'write_file',
     'list_directory']



You may activate tools on a per-invokation basis by simply passing `tools` or `toolkits` as arguements to the chat functions

## Chat History
While the LLM itself is just a function, we build conversation-like behavior by storing a chat history. In `simplechatbot`, the history is stored in a `ChatHistory`, which is just a list subtype where list elements contain langchain `BaseMessage` subtypes. You can access it through the `history` property, and work with it just as a list.


```python
chatbot.history
```




    []



To see the conversation history that is sent to the LLM, you can use the `get_buffer_string` method. This uses the same underlying functions as langchain so you can use this for debugging.


```python
print(chatbot.history.get_buffer_string())
```

    


## Chat Methods

The methods `chat` and `chat_stream` both send the provided message with history to the LLM for prediction.

+ `chat`: returns `ChatResult` instance with the AI message and ability to call tool functions with provided parameters.
+ `chat_stream`: returns `ChatStreamResult` instance which allows you to iterate over responses from the LLM and call tools once all of the parameters have been provided.

I will now describe them in more detail.


```python
system_prompt = '''
Your job is to answer any questions the user has.
'''
chatbot = simplechatbot.chatbot.ChatBot.from_model(
    model = openai_model,
    system_prompt=system_prompt,
    tools=tools,
)

reply = chatbot.chat('What is the capital of France?')
reply.message.content
```




    'The capital of France is Paris.'



### `ChatResult` Objects

These objects are returned from calls to `chat`. Most importantly, access the `message` property to see the response directly.


```python
reply.message.usage_metadata
```




    {'input_tokens': 328,
     'output_tokens': 8,
     'total_tokens': 336,
     'input_token_details': {'cache_read': 0},
     'output_token_details': {'reasoning': 0}}




```python
reply.message.response_metadata
```




    {'token_usage': {'completion_tokens': 8,
      'prompt_tokens': 328,
      'total_tokens': 336,
      'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0},
      'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}},
     'model_name': 'gpt-4o-mini-2024-07-18',
     'system_fingerprint': 'fp_f59a81427f',
     'finish_reason': 'stop',
     'logprobs': None}



See available tool calls using `tool_calls`. This is a thin wrapper over the `AIMessage.tool_calls` property.

See how the question about the capital of France does not require a tool call but the request to save a note does require a tool call.


```python
chatbot.chat('What is the capital of France?').tool_calls
```




    []




```python
reply = chatbot.chat('Can you save a note to the file "note.txt" that says "Hello, world!"?')
reply.tool_calls
```




    [{'name': 'write_file',
      'args': {'file_path': 'note.txt', 'text': 'Hello, world!'},
      'id': 'call_jRYzkpAN1PmpGycoZI0LTKdu',
      'type': 'tool_call'}]



Make a call to `call_tools()` to actually execute all of the tools. The results are returned as dicts with tool names as keys pointing to `ToolCallResult` instances, which contain the tool call id, the tool call arguments, a reference to the tool object, and the value returned from the tool, among other things.


```python
tool_call_results = reply.call_tools()
for tool_name, result in tool_call_results.items():
    print(result.tool_info_str(), '->', result.return_value)
    print(result)
```

    write_file(file_path=note.txt, text=Hello, world!) -> File written successfully to note.txt.
    ToolCallResult(id='call_jRYzkpAN1PmpGycoZI0LTKdu', name='write_file', type='tool_call', args={'file_path': 'note.txt', 'text': 'Hello, world!'}, tool=WriteFileTool(root_dir='/tmp/tmp42vq4a65'), return_value='File written successfully to note.txt.')


And we can verify that the LLM actually wrote the file.


```python
import os
os.listdir(working_directory.name)
```




    ['note.txt']



### `ChatStreamResult` Objects

Calls to `chat_stream` are very similar but instead return `ChatStreamResult` objects. These objects are iterators and must be iterated over before you can call any tools. If you are not expecting any tool calls, you can call `chat_stream` directly in the loop.


```python
for r in chatbot.chat_stream(f'What is the capital of France?'):
    print(r.content, end='', flush=True)
```

    The capital of France is Paris.

If you want to be able to call tools while streaming, iterate through the reply chunks to completion before executing the tool calls using `call_tools`. In the code below we receive the stream object with the original call to `chat_stream`, iterate through the reply content, and then, when the iterable has been exhausted, call `call_tools`.


```python
reply_stream = chatbot.chat_stream(f'Can you save a note to the file "hello.txt" that says "You put me here as a test!"?')
for r in reply_stream:
    print(r.content, end='', flush=True)

reply_stream.tool_calls
```




    [{'name': 'write_file',
      'args': {'file_path': 'hello.txt', 'text': 'You put me here as a test!'},
      'id': 'call_XDr0sxLMpU92wllgUWpIDQf6',
      'type': 'tool_call'}]




```python
reply_stream.call_tools()
```




    {'write_file': ToolCallResult(id='call_XDr0sxLMpU92wllgUWpIDQf6', name='write_file', type='tool_call', args={'file_path': 'hello.txt', 'text': 'You put me here as a test!'}, tool=WriteFileTool(root_dir='/tmp/tmp42vq4a65'), return_value='File written successfully to hello.txt.')}



Note: do not call `call_tools` before completing the iterable because the chatbot may not have streamed all of the tool call information back.

## Chat User Interface
Of course, what is a chatbot if you can't actually use it? To run an interactive command-line chat, use `.ui.start_interactive`.


```python
# uncomment to start interactive chat
#chatbot.ui.start_interactive(stream=True, show_intro=True, show_tools=True)
```
