


# API Introduction
This is a brief introduction to the ```simplechatbot``` API.




---

``` python linenums="1"
import sys
sys.path.append('../src/')

import simplechatbot
```


---




## Instantiating `Agent` Objects

`Agent` instances maintain three elements: a chat model (or runnable) LLM, chat history, and available tools / functions. System prompts are simply stored as part of the conversation history.

It may be instantiated from any [langchain chat model](https://python.langchain.com/v0.1/docs/modules/model_io/chat/) or runnable.




---

``` python linenums="1"
from langchain_openai import ChatOpenAI

# optional: use this to grab keys from a json file rather than setting system variables
keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')

openai_model = ChatOpenAI(model='gpt-4o-mini', api_key=keychain['openai'])
agent = simplechatbot.Agent.from_model(model=openai_model)
print(agent)
```



stdout:
 

    Agent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={}))
    

 



---




Set a system prompt for the agent by passing it as the `system_prompt` argument.




---

``` python linenums="1"
system_prompt = '''
You are a creative designer who has been tasked with creating a new slogan for a company.
The user will describe the company, and you will need to generate three slogan ideas for them.
'''
agent = simplechatbot.Agent.from_model(
    model = openai_model,
    system_prompt=system_prompt,
)
```


---




The `tools` parameter allows you to pass any [langchain tools](https://python.langchain.com/v0.1/docs/modules/tools/) you want your agent to be able to use. You can use one of [Langchain's built-in tools](https://python.langchain.com/v0.1/docs/integrations/tools/) (such as `FileManagementToolkit`) or [define your own custom tools](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/). I will use `FileManagementToolkit` for demonstration purposes here.




---

``` python linenums="1"
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


---




You can see that tools are added to an internal `ToolSet` object.




---

``` python linenums="1"
agent.toolset
```




text:

    ToolSet(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)}, tool_factories=[], tool_choice=None)
 


 


 



---




While the LLM itself is just a function, we build conversation-like behavior by storing a chat history. In `simplechatbot`, the history is stored in a `ChatHistory`, which is just a list subtype where list elements contain langchain `BaseMessage` subtypes. You can access it through the `history` property, and work with it just as a list.

Here you can see that the system prompt is simply added as the first message in the agent history. 




---

``` python linenums="1"
agent.history
```




text:

    []
 


 


 



---




To see the conversation history that is sent to the LLM, you can use the `get_buffer_string` method. This uses the same langchain methods used to invoke the LLM, so it is useful for debugging.




---

``` python linenums="1"
print(agent.history.get_buffer_string())
```



stdout:
 

    
    

 



---




Note that history is a `list` subtype, so you can iterate through messages as you would expect.




---

``` python linenums="1"
for m in agent.history:
    print(m)
```


---




## High-level `chat` and `stream` Methods

There are two primary methods used to interact with the chatbot: `chat` and `stream`. 

These are the method use-cases:

`.chat()` → `ChatResult`: Use when you want to retrieve the full LLM response at once when it finishes.

`.stream()` → `ChatStream`: Use when you would like to show intermediary results to the user as they are received from the LLM.




---

``` python linenums="1"
agent.chat('My name is Devin.')
```




text:

    ChatResult(content=Nice to meet you, Devin! How can I assist you today?, tool_calls=[])
 


 


 



---





---

``` python linenums="1"
agent.stream('My name is Devin and I am a creative designer.')
```




text:

    StreamResult(message_iter=<generator object RunnableBindingBase.stream at 0x131542d40>, agent=Agent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)})), tool_lookup=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)}), add_reply_to_history=True, full_message=AIMessageChunk(content='', additional_kwargs={}, response_metadata={}), exhausted=False, receive_callback=None)
 


 


 



---




Again use the `get_buffer_string` method to conveniently view the chat history.




---

``` python linenums="1"
print(agent.history.get_buffer_string())
```



stdout:
 

    Human: My name is Devin.
    AI: Nice to meet you, Devin! How can I assist you today?
    Human: My name is Devin and I am a creative designer.
    

 



---




From the response to the prompt below you can see that it is maintained in the chat history because it "retains" knowledge that is given to it.




---

``` python linenums="1"
agent.chat('I have a quiz for you: what is my name?')
```




text:

    ChatResult(content=Your name is Devin!, tool_calls=[])
 


 


 



---




#### `.chat()` and `ChatResult` Objects

The `chat` method submits the current message and all history to the LLM and returns the reply as a `ChatResult` object.




---

``` python linenums="1"
agent.chat('Hello world.')
```




text:

    ChatResult(content=Hello, Devin! How are you today?, tool_calls=[])
 


 


 



---




If you want to submit the current chat history but do not want to add a new message, you can pass `None` as the message argument.




---

``` python linenums="1"
agent.chat(None)
```




text:

    ChatResult(content=Is there anything specific you'd like to discuss or any questions you have?, tool_calls=[])
 


 


 



---




Alternatively, if you want to submit a query to the LLM but do not want to save it in the history, set `add_to_history = False`.




---

``` python linenums="1"
agent.chat('Hello world.', add_to_history=False)
```




text:

    ChatResult(content=Hello again, Devin! If there's anything specific you'd like to talk about or if you have any questions, feel free to share!, tool_calls=[])
 


 


 



---




`ChatResult` objects are returned from `chat()` and `invoke()` calls and include the LLM response text or tool calling information.




---

``` python linenums="1"
result = agent.chat('What is my name?')
result
```




text:

    ChatResult(content=Your name is Devin., tool_calls=[])
 


 


 



---




If no tool calls were requested from the LLM, you can access the response as a string through the `content` property.




---

``` python linenums="1"
result.content
```




text:

    'Your name is Devin.'
 


 


 



---




If tool calls were made, the content will be empty but you can get information about any tool calls through the `tool_calls` attribute. Notice that no tool calls were requested by the LLM in the response to this query.




---

``` python linenums="1"
result.tool_calls
```




text:

    []
 


 


 



---




If there were tool calls, you can execute them using the `execute_tools` method.




---

``` python linenums="1"
result.execute_tools()
```




text:

    {}
 


 


 



---




We provided the agent with a tool called `check_new_messages` earlier, and the LLM will request a tool call if the user requests it.




---

``` python linenums="1"
result = agent.chat('Check new messages.')
result.tool_calls
```




text:

    [ToolCallInfo(id='call_208aNIO1I1TlXxijnOdWYQFf', name='check_new_messages', type='tool_call', args={'text': 'Check new messages.', 'username': 'Devin'}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>))]
 


 


 



---




The `execute_tools` method returns a dictionary of `ToolCallResult` objects which contain the tool call information from the LLM (`ToolCallInfo`) and the return value of the tool execution.




---

``` python linenums="1"
tool_results = result.execute_tools()
tool_results
```




text:

    {'check_new_messages': ToolCallResult(info=ToolCallInfo(id='call_208aNIO1I1TlXxijnOdWYQFf', name='check_new_messages', type='tool_call', args={'text': 'Check new messages.', 'username': 'Devin'}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)), return_value='No new messages.')}
 


 


 



---




Use the `return_value` attribute to access these results.




---

``` python linenums="1"
tool_results['check_new_messages'].return_value
```




text:

    'No new messages.'
 


 


 



---




#### `.stream()` and `StreamResult` Objects

`stream` is very similar to `chat`, but allows you to return content to the user as soon as the LLM produces it. The method returns a `StreamResult` object which has an iterator interface that accumulates results from the LLM while also returning incremental results.

In this example, I call `stream` to retrieve a `StreamResult` object, which I then iterate through to retrieve and print all results.




---

``` python linenums="1"
stream = agent.stream('What is my name?')
for r in stream:
    print(r.content, end='', flush=True)
stream
```



stdout:
 

    Your name is Devin.

 





text:

    StreamResult(message_iter=<generator object RunnableBindingBase.stream at 0x1229117b0>, agent=Agent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)})), tool_lookup=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)}), add_reply_to_history=True, full_message=AIMessageChunk(content='Your name is Devin.', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_06737a9306'}), exhausted=True, receive_callback=None)
 


 


 



---




You can check the `exhausted` flag to see if the LLM has returned all results yet.




---

``` python linenums="1"
stream = agent.stream('What is my name?')
print(stream.exhausted)
for r in stream:
    print(r.content, end='', flush=True)
print(stream.exhausted)
stream
```



stdout:
 

    False
    Your name is Devin.True
    

 





text:

    StreamResult(message_iter=<generator object RunnableBindingBase.stream at 0x1315422f0>, agent=Agent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)})), tool_lookup=ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)}), add_reply_to_history=True, full_message=AIMessageChunk(content='Your name is Devin.', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_06737a9306'}), exhausted=True, receive_callback=None)
 


 


 



---




After retrieving all of the LLM response, you can check if any tool calls are required.




---

``` python linenums="1"
stream = agent.stream('Check my messages.')
for r in stream:
    print(r.content, end='', flush=True)
stream.tool_calls
```




text:

    [ToolCallInfo(id='call_wxhHcOtuj5XPSHvnbFVvOXrQ', name='check_new_messages', type='tool_call', args={'text': 'Check my messages.', 'username': 'Devin'}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>))]
 


 


 



---




And you would similarly execute tools by calling `execute_tools`. Note that you cannot call this method if the stream has not been exhausted.




---

``` python linenums="1"
stream.execute_tools()
```




text:

    {'check_new_messages': ToolCallResult(info=ToolCallInfo(id='call_wxhHcOtuj5XPSHvnbFVvOXrQ', name='check_new_messages', type='tool_call', args={'text': 'Check my messages.', 'username': 'Devin'}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x131f511c0>)), return_value='No new messages.')}
 


 


 



---




You can use the `result` method to get a `ChatResult` object instead. If it has not retrieved all results from the LLM, it will do so before returning.




---

``` python linenums="1"
agent.stream('Hello world.').collect()
```




text:

    ChatResult(content=Hello again, Devin! What would you like to talk about?, tool_calls=[])
 


 


 



---




## Low-level LLM Methods: `_invoke` and `_stream`

These lower-level `_invoke` and `_stream` methods are used by the `chat` and `chat_stream` methods to submit prompts to the LLM. They can allow you to interact with the LLM and tools/functions without chat history. Their signatures are very similar to high-level methods and they return the same types.

***NOTE***: *These methods ignore the system prompt!*

The low-level `_invoke` method returns a `ChatResult` object with the content and tool call information.




---

``` python linenums="1"
result = agent._invoke('Hello world!')
result
```




text:

    ChatResult(content=Hello! How can I assist you today?, tool_calls=[])
 


 


 



---




And `stream` is very similar to `stream` except that it ignores chat history.




---

``` python linenums="1"
stream = agent._stream('Check messages.')
for r in stream:
    print(r.content, end='', flush=True)
stream.execute_tools()
```



stdout:
 

    Can you please provide your username and the message you would like to check?

 





text:

    {}
 


 


 



---




## Chat User Interface
Of course, what is a chatbot if you can't actually use it? To run an interactive command-line chat, use `.ui.start_interactive`.




---

``` python linenums="1"
# uncomment to start interactive chat
#agent.ui.start_interactive(stream=True, show_intro=True, show_tools=True)
```


---


 