# Introduction
This is a brief introduction to the ```simplechatbot``` package.


```python
import sys
sys.path.append('../src/')

import simplechatbot
```

## Instantiating `Agent` Objects

`Agent` instances maintain three elements: a chat model (or runnable) LLM, chat history, and available tools / functions.

It may be instantiated from any [langchain chat model](https://python.langchain.com/v0.1/docs/modules/model_io/chat/) or runnable.


```python
from langchain_openai import ChatOpenAI

# optional: use this to grab keys from a json file rather than setting system variables
keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')

openai_model = ChatOpenAI(model='gpt-4o-mini', api_key=keychain['openai'])
agent = simplechatbot.Agent.from_model(model=openai_model)
print(agent)
```

    Agent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={}))


The `tools` parameter allows you to pass any [langchain tools](https://python.langchain.com/v0.1/docs/modules/tools/) you want your agent to be able to use. You can use one of [Langchain's built-in tools](https://python.langchain.com/v0.1/docs/integrations/tools/) (such as `FileManagementToolkit`) or [define your own custom tools](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/). I will use `FileManagementToolkit` for demonstration purposes here.


```python
import langchain_core.tools

@langchain_core.tools.tool
def check_new_messages(text: str, username: str) -> str:
    '''Check messages.'''
    return f'No new messages.'

agent = simplechatbot.Agent.from_model(
    model = openai_model,
    tools = [check_new_messages],
)
```

You can see that tools are added to an internal `ToolSet` object.


```python
agent.toolset
```




    ToolSet(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)}, tool_factories=[], tool_choice=None)



Set a system prompt for the agent by passing it as the `system_prompt` argument.


```python
system_prompt = '''
You are a creative designer who has been tasked with creating a new slogan for a company.
The user will describe the company, and you will need to generate three slogan ideas for them.
'''
agent = simplechatbot.Agent.from_model(
    model = openai_model,
    tools = [check_new_messages],
    system_prompt=system_prompt,
)
```

While the LLM itself is just a function, we build conversation-like behavior by storing a chat history. In `simplechatbot`, the history is stored in a `ChatHistory`, which is just a list subtype where list elements contain langchain `BaseMessage` subtypes. You can access it through the `history` property, and work with it just as a list.

Here you can see that the system prompt is simply added as the first message in the agent history. 


```python
agent.history
```




    [SystemMessage(content='\nYou are a creative designer who has been tasked with creating a new slogan for a company.\nThe user will describe the company, and you will need to generate three slogan ideas for them.\n', additional_kwargs={}, response_metadata={})]



To see the conversation history that is sent to the LLM, you can use the `get_buffer_string` method. This uses the same langchain methods used to invoke the LLM, so it is useful for debugging.


```python
print(agent.history.get_buffer_string())
```

    System: 
    You are a creative designer who has been tasked with creating a new slogan for a company.
    The user will describe the company, and you will need to generate three slogan ideas for them.
    


Note that history is a `list` subtype, so you can iterate through messages as you would expect.


```python
for m in agent.history:
    print(m)
```

    content='\nYou are a creative designer who has been tasked with creating a new slogan for a company.\nThe user will describe the company, and you will need to generate three slogan ideas for them.\n' additional_kwargs={} response_metadata={}


## High-level `chat` and `stream` Methods

There are two primary methods used to interact with the chatbot: `chat` and `stream`. 

These are the method use-cases:

`.chat()` → `ChatResult`: Use when you want to retrieve the full LLM response at once when it finishes.

`.stream()` → `ChatStream`: Use when you would like to show intermediary results to the user as they are received from the LLM.


```python
agent.chat('My name is Devin.')
```




    ChatResult(content=Nice to meet you, Devin! How can I assist you today? If you have a company description, I can help create some catchy slogans for you!, tool_calls=[])




```python
agent.stream('My name is Devin and I am a creative designer.')
```




    StreamResult(message_iter=<generator object RunnableBindingBase.stream at 0x11632e890>, agent=Agent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)})), tool_lookup=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)}), add_reply_to_history=True, full_message=AIMessageChunk(content='', additional_kwargs={}, response_metadata={}), exhausted=False, receive_callback=None)



Again use the `get_buffer_string` method to conveniently view the chat history.


```python
print(agent.history.get_buffer_string())
```

    System: 
    You are a creative designer who has been tasked with creating a new slogan for a company.
    The user will describe the company, and you will need to generate three slogan ideas for them.
    
    Human: My name is Devin.
    AI: Nice to meet you, Devin! How can I assist you today? If you have a company description, I can help create some catchy slogans for you!
    Human: My name is Devin and I am a creative designer.


From the response to the prompt below you can see that it is maintained in the chat history because it "retains" knowledge that is given to it.


```python
agent.chat('I have a quiz for you: what is my name?')
```




    ChatResult(content=Your name is Devin!, tool_calls=[])



#### `.chat()` and `ChatResult` Objects

The `chat` method submits the current message and all history to the LLM and returns the reply as a `ChatResult` object.


```python
agent.chat('Hello world.')
```




    ChatResult(content=Hello, Devin! How can I assist you today?, tool_calls=[])



If you want to submit the current chat history but do not want to add a new message, you can pass `None` as the message argument.


```python
agent.chat(None)
```




    ChatResult(content=If you're looking for creative ideas or need assistance with something specific, feel free to let me know!, tool_calls=[])



Alternatively, if you want to submit a query to the LLM but do not want to save it in the history, set `add_to_history = False`.


```python
agent.chat('Hello world.', add_to_history=False)
```




    ChatResult(content=Hello again! If you have any questions or something specific you’d like to discuss, just let me know!, tool_calls=[])



`ChatResult` objects are returned from `chat()` and `invoke()` calls and include the LLM response text or tool calling information.


```python
result = agent.chat('What is my name?')
result
```




    ChatResult(content=Your name is Devin!, tool_calls=[])



If no tool calls were requested from the LLM, you can access the response as a string through the `content` property.


```python
result.content
```




    'Your name is Devin!'



If tool calls were made, the content will be empty but you can get information about any tool calls through the `tool_calls` attribute. Notice that no tool calls were requested by the LLM in the response to this query.


```python
result.tool_calls
```




    []



If there were tool calls, you can execute them using the `execute_tools` method.


```python
result.execute_tools()
```




    {}



We provided the agent with a tool called `check_new_messages` earlier, and the LLM will request a tool call if the user requests it.


```python
result = agent.chat('Check new messages.')
result.tool_calls
```




    [ToolCallInfo(id='call_WhqKp8uQZW882oC0OPBbDtsl', name='check_new_messages', type='tool_call', args={'text': 'Check new messages.', 'username': 'Devin'}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>))]



The `execute_tools` method returns a dictionary of `ToolCallResult` objects which contain the tool call information from the LLM (`ToolCallInfo`) and the return value of the tool execution.


```python
tool_results = result.execute_tools()
tool_results
```




    {'check_new_messages': ToolCallResult(info=ToolCallInfo(id='call_WhqKp8uQZW882oC0OPBbDtsl', name='check_new_messages', type='tool_call', args={'text': 'Check new messages.', 'username': 'Devin'}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)), return_value='No new messages.')}



Use the `return_value` attribute to access these results.


```python
tool_results['check_new_messages'].return_value
```




    'No new messages.'



#### `.stream()` and `StreamResult` Objects

`stream` is very similar to `chat`, but allows you to return content to the user as soon as the LLM produces it. The method returns a `StreamResult` object which has an iterator interface that accumulates results from the LLM while also returning incremental results.

In this example, I call `stream` to retrieve a `StreamResult` object, which I then iterate through to retrieve and print all results.


```python
stream = agent.stream('What is my name?')
for r in stream:
    print(r.content, end='', flush=True)
stream
```

    Your name is Devin!




    StreamResult(message_iter=<generator object RunnableBindingBase.stream at 0x115a398a0>, agent=Agent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)})), tool_lookup=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)}), add_reply_to_history=True, full_message=AIMessageChunk(content='Your name is Devin!', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_06737a9306'}), exhausted=True, receive_callback=None)



You can check the `exhausted` flag to see if the LLM has returned all results yet.


```python
stream = agent.stream('What is my name?')
print(stream.exhausted)
for r in stream:
    print(r.content, end='', flush=True)
print(stream.exhausted)
stream
```

    False
    Your name is Devin!True





    StreamResult(message_iter=<generator object RunnableBindingBase.stream at 0x11632dd50>, agent=Agent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)})), tool_lookup=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)}), add_reply_to_history=True, full_message=AIMessageChunk(content='Your name is Devin!', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_06737a9306'}), exhausted=True, receive_callback=None)



After retrieving all of the LLM response, you can check if any tool calls are required.


```python
stream = agent.stream('Check my messages.')
for r in stream:
    print(r.content, end='', flush=True)
stream.tool_calls
```




    [ToolCallInfo(id='call_HDH2aVaLMWzJF8jDnzUioDhi', name='check_new_messages', type='tool_call', args={'text': 'Check my messages.', 'username': 'Devin'}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>))]



And you would similarly execute tools by calling `execute_tools`. Note that you cannot call this method if the stream has not been exhausted.


```python
stream.execute_tools()
```




    {'check_new_messages': ToolCallResult(info=ToolCallInfo(id='call_HDH2aVaLMWzJF8jDnzUioDhi', name='check_new_messages', type='tool_call', args={'text': 'Check my messages.', 'username': 'Devin'}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x116d2f560>)), return_value='No new messages.')}



You can use the `result` method to get a `ChatResult` object instead. If it has not retrieved all results from the LLM, it will do so before returning.


```python
agent.stream('Hello world.').collect()
```




    ChatResult(content=Hello again, Devin! How can I help you today?, tool_calls=[])



## Low-level LLM Methods: `_invoke` and `_stream`

These lower-level `_invoke` and `_stream` methods are used by the `chat` and `chat_stream` methods to submit prompts to the LLM. They can allow you to interact with the LLM and tools/functions without chat history. Their signatures are very similar to high-level methods and they return the same types.

***NOTE***: *These methods ignore the system prompt!*

The low-level `_invoke` method returns a `ChatResult` object with the content and tool call information.


```python
result = agent._invoke('Hello world!')
result
```




    ChatResult(content=Hello! How can I assist you today?, tool_calls=[])



And `stream` is very similar to `stream` except that it ignores chat history.


```python
stream = agent._stream('Check messages.')
for r in stream:
    print(r.content, end='', flush=True)
stream.execute_tools()
```

    Could you please provide your username and the specific text you would like me to check for new messages?




    {}



## Chat User Interface
Of course, what is a chatbot if you can't actually use it? To run an interactive command-line chat, use `.ui.start_interactive`.


```python
# uncomment to start interactive chat
#agent.ui.start_interactive(stream=True, show_intro=True, show_tools=True)
```
