# Overview
Brief overview of using ```simplechatbot``` package.


```python
import sys
sys.path.append('..')

import simplechatbot.v4 as simplechatbot
```

## `ChatBot` Objects

`ChatBot` instances maintain the three things that define a chatbot: a chat model (or runnable), chat history, and available tools / functions.

It may be created from any [langchain chat model](https://python.langchain.com/v0.1/docs/modules/model_io/chat/) or runnable. For convenience, you may also instantiate directly from Ollama or 

Optional: you can use `APIKeyChain` to retrive API keys from a json file. It is a `dict` subclass with the `from_json_file` method that will simply read a json file as a dict.


```python
from langchain_openai import ChatOpenAI

keychain = simplechatbot.APIKeyChain.from_json_file('../scripts/keys.json')

model = ChatOpenAI(model='gpt-4o-mini', api_key=keychain['openai'])
chatbot = simplechatbot.ChatBot.from_model(model=model)
print(chatbot)
```

    ChatBot(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=None)


If you have an ollama instance running on your machine, you can use the `from_ollama` constructor. You only need the model name and the `ChatOllma` constructor will identify the model API endpoint location as long as it is local.


```python
chatbot = simplechatbot.ChatBot.from_ollama(
    model_name = 'llama3.1', 
)
```

You may do the same for `ChatOpenAI` using `from_openai`.


```python
chatbot = simplechatbot.ChatBot.from_openai(
    model_name = 'gpt-4o-mini', 
    api_key=keychain['openai'],
)
```

Use the `system_prompt` parameter to initialize the chatbot with instructions.


```python
system_prompt = '''
You are a creative designer who has been tasked with creating a new slogan for a company.
The user will describe the company, and you will need to generate three slogan ideas for them.
'''
chatbot = simplechatbot.ChatBot.from_ollama(
    model_name = 'llama3.1', 
    system_prompt=system_prompt,
)
```

The `tools` parameter allows you to pass any [langchain tools](https://python.langchain.com/v0.1/docs/modules/tools/) you want your chatbot to be able to use. You can use one of [Langchain's built-in tools](https://python.langchain.com/v0.1/docs/integrations/tools/) (such as `DuckDuckGoSearchResults`) or [define your own custom tools](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/).


```python
from langchain_community.tools import DuckDuckGoSearchResults

tools = [
    DuckDuckGoSearchResults(
        keys_to_include=['snippet', 'title'], 
        results_separator='\n\n',
        num_results = 4,
    ),
]

chatbot = simplechatbot.ChatBot.from_ollama(
    model_name = 'llama3.1', 
    tools=tools,
)
```

Having a chat.


```python
system_prompt = '''
Your job is to answer any questions the user has.
'''
chatbot = simplechatbot.ChatBot.from_openai(
    model_name = 'gpt-4o-mini', 
    api_key=keychain['openai'],
    system_prompt=system_prompt,
    tools=tools,
)

reply = chatbot.chat('What is the capital of France?')
reply.message.content
```




    'The capital of France is Paris.'



## `ChatResult` and `ChatStream` Objects

Calls to `chat` return `ChatResult` objects, which contain the message reply (`message`) and means with which to call any tools (`call_tools`).


```python
reply.message.usage_metadata
```




    {'input_tokens': 104, 'output_tokens': 8, 'total_tokens': 112}




```python
reply.message.response_metadata
```




    {'token_usage': {'completion_tokens': 8,
      'prompt_tokens': 104,
      'total_tokens': 112,
      'completion_tokens_details': {'reasoning_tokens': 0},
      'prompt_tokens_details': {'cached_tokens': 0}},
     'model_name': 'gpt-4o-mini-2024-07-18',
     'system_fingerprint': 'fp_f85bea6784',
     'finish_reason': 'stop',
     'logprobs': None}



See available tool calls using `tool_calls`. See how the question about the weather in Rome results in a tool call.


```python
chatbot.chat('What is the capital of France?').tool_calls
```




    []




```python
reply = chatbot.chat('What is the weather in Rome?')
reply.tool_calls
```




    [{'name': 'duckduckgo_results_json',
      'args': {'query': 'current weather in Rome'},
      'id': 'call_wF7eMhgRjPAR3CNmlXQyuJZW',
      'type': 'tool_call'}]




```python
for tool_name, result in reply.call_tools().items():
    print(result.tool_info_str)
```

    duckduckgo_results_json(query=current weather in Rome)


Calls to `chat_stream` are very similar but instead return `ChatStream` objects. These objects are iterators and must be iterated over before you can call any tools.


```python
for r in chatbot.chat_stream(f'What is the capital of France?'):
    print(r.content, end='', flush=True)
```

    The capital of France is Paris.

After exhausting the iterable, you can call `tool_calls` and `call_tools` to see and execute the requested tools.


```python
reply_stream = chatbot.chat_stream(f'What is the weather in Rome?')
for r in reply_stream:
    print(r.content, end='', flush=True)

reply_stream.tool_calls
```

    The current weather in Rome is sunny, with a few clouds expected. The temperature is peaking at around 79°F. During the night and early morning, there will be light air, and in the afternoon, a light breeze is anticipated, with gusts possibly reaching up to 21 mph.




    []



Note: do not call `call_tools` before completing the iterable because the chatbot may not have streamed all of the tool call information back.

## Chat User Interface
Of course, what is a chatbot if you can't actually use it? To run an interactive command-line chat, use `.ui.start_interactive`.


```python
# uncomment to start interactive chat
#chatbot.ui.start_interactive(stream=True, show_intro=True, show_tools=True)
```
