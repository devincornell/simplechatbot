# Tool Calling

`simplechatbot` empowers chatbot agents with the ability to produce arguments for arbitrary user functions instead of providing a text response to the user's prompt. Using this interface you can enable features such as web searching, email sending/checking, file browsing, image creation, or any other functionality that can be accessed through Python. The LLM will "decide" whether and which tools/functions should be executed based on a given prompt, so the key is to use tools with clear and concise instructions.

Under the hood, `ChatBot` instances maintain a collection of [langchain tools](https://python.langchain.com/docs/how_to/#tools) which can be extracted from toolkits or even factory methods that accept the chatbot itself as a parameter. Tools may also be added at the time of LLM execution to enable dynamic systems of available tools.

You can create your own [custom tools](https://python.langchain.com/docs/how_to/custom_tools/) or choose from [Langchain's built-in tools](https://python.langchain.com/docs/integrations/tools/). I will use `FileManagementToolkit` for demonstration purposes here.


```python
import sys
sys.path.append('..')

import simplechatbot
from simplechatbot.openai import OpenAIChatBot
```

## Enabling Tools
Start by creating a new example tool that can enables the LLM to check email for the user. We create this tool using the `@langchain_core.tools.tool` decorator.


```python
import langchain_core.tools

@langchain_core.tools.tool
def check_new_messages() -> str:
    '''Check messages.'''
    return f'No new messages.'
```

We include this tool as part of the chatbot by passing the function through the `tools` argument.


```python
keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')

system_prompt = '''
You are designed to answer any question the user has and send/check messages if needed.
When the user requests you to check your messages, you should display the retrieved messages
 to the user.
'''

chatbot = OpenAIChatBot.new(
    model_name = 'gpt-4o-mini', 
    api_key=keychain['openai'],
    system_prompt=system_prompt,
    tools = [check_new_messages],
)
```

Now the LLM will have access to these tools. While the chatbot instance stores the LLM object in the `_model` attribute, you can use `model` to get the LLM with bound tools.


```python
chatbot.model
```




    RunnableBinding(bound=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x128006a80>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x1280306e0>, root_client=<openai.OpenAI object at 0x10f153a40>, root_async_client=<openai.AsyncOpenAI object at 0x128006ae0>, model_name='gpt-4o-mini', model_kwargs={}, openai_api_key=SecretStr('**********')), kwargs={'tools': [{'type': 'function', 'function': {'name': 'check_new_messages', 'description': 'Check messages.', 'parameters': {'properties': {}, 'type': 'object'}}}]}, config={}, config_factories=[])



You can also use the method `get_model_with_tools` to get the tool-bound model with any additional tools. The `invoke`, `stream`, `chat`, and `chat_stream` methods all use this under-the hood so you can add any tools, toolkits, or tool factories to the model at invokation.


```python
chatbot.get_model_with_tools(tools=None)
```




    (RunnableBinding(bound=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x128006a80>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x1280306e0>, root_client=<openai.OpenAI object at 0x10f153a40>, root_async_client=<openai.AsyncOpenAI object at 0x128006ae0>, model_name='gpt-4o-mini', model_kwargs={}, openai_api_key=SecretStr('**********')), kwargs={'tools': [{'type': 'function', 'function': {'name': 'check_new_messages', 'description': 'Check messages.', 'parameters': {'properties': {}, 'type': 'object'}}}]}, config={}, config_factories=[]),
     ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x10f957ce0>)}))



Tools will be automatically used when we call any of the invoke or stream methods.

Notice that the LLM behaves normally if the user's prompts are unrelated to the tool.


```python
chatbot.invoke('Hello world!')
```




    ChatResult(content=Hello! How can I assist you today?, tool_calls=[])



If the LLM "decides" that the user needs to execute a tool, it returns a tool call as the response instead of returning content.


```python
result = chatbot.invoke('Check my messages.')
result
```




    ChatResult(content=, tool_calls=[ToolCallInfo(id='call_ngajeroEwfXipobUvG6sPNxd', name='check_new_messages', type='tool_call', args={}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x10f957ce0>))])



The tool call information can be accessed through the `ChatResult.tool_calls` attribute, which is simply a list supertype. Use `tool_info_str` to clearly show the arguments being passed to the function.


```python
for tc in result.tool_calls:
    print(tc.tool_info_str())
```

    check_new_messages()


You may also provide additional tools at the time of invoking the LLM, and it will be treated as if it was part of the chatbot. 

In this example, we create a new tool with two arguments that must be provided by the LLM.


```python
@langchain_core.tools.tool
def send_message(recipient: str, text: str) -> str:
    '''Send messages to others.'''
    return f'Message sent!'

result = chatbot.invoke('Send a message to Bob saying "Hello!"', tools=[send_message])
result
```




    ChatResult(content=, tool_calls=[ToolCallInfo(id='call_p4znAVb29LTFT6TF8qnkG1NU', name='send_message', type='tool_call', args={'recipient': 'Bob', 'text': 'Hello!'}, tool=StructuredTool(name='send_message', description='Send messages to others.', args_schema=<class 'langchain_core.utils.pydantic.send_message'>, func=<function send_message at 0x12804bb00>))])



You can see that the LLM provided the `recipient` and `text` arguments which were passed to the function call information.


```python
result.tool_calls[0].tool_info_str()
```




    'send_message(recipient=Bob, text=Hello!)'



You can adjust behavior using the `tool_choice` argument in the chatbot constructor or at invokation. The value `'any'` means that a tool MUST be called, but all tools are candidates. The value `'auto'` (the default) allows the LLM to reply with normal content rather than a tool call, and you can also pass the name of a specific function as well.


```python
result = chatbot.invoke('Go to the store for me!', tool_choice='any')
result.tool_calls[0].tool_info_str()
```




    'check_new_messages()'



## Executing Tools
Tools allow the LLM to determine if and when to execute tools and also provides parameters for the tool call based on conversation history, but the user containing function is responsible for actually executing the tool with the arguments from the LLM.

Use the `execute_tools` method to actually execute the tool, which returns a mapping of tool names to `ToolCallResult` objects.


```python
result = chatbot.invoke('Check my messages.')
result.tool_calls[0].tool_info_str()
```




    'check_new_messages()'




```python
tr = result.execute_tools()
tr
```




    {'check_new_messages': ToolCallResult(info=ToolCallInfo(id='call_sOvFP85Zw090E9xin3c74uCl', name='check_new_messages', type='tool_call', args={}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x10f957ce0>)), return_value='No new messages.')}



Get the return value from the tool through the `return_value` property.


```python
tr['check_new_messages'].return_value
```




    'No new messages.'



Extracting tool calls from a `StreamResult` is a little more complicated because the stream must be exhausted before executing tools. This happens because the tool call information replaces the text response, so the streamer is essentially receiving chunks of the function call information until exhaustion.

The calling function must handle both the streamed output and tool calls.


```python
stream = chatbot.stream('Check my messages.')
for r in stream:
    print(r, end='', flush=True)
if len(stream.tool_calls) > 0:
    stream.execute_tools()
```

    content='' additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_8MqiTmWRcqC9xjVUqzY1sq2P', 'function': {'arguments': '', 'name': 'check_new_messages'}, 'type': 'function'}]} response_metadata={} id='run-825b8400-30dd-4d02-ac24-47d0cc9c2cac' tool_calls=[{'name': 'check_new_messages', 'args': {}, 'id': 'call_8MqiTmWRcqC9xjVUqzY1sq2P', 'type': 'tool_call'}] tool_call_chunks=[{'name': 'check_new_messages', 'args': '', 'id': 'call_8MqiTmWRcqC9xjVUqzY1sq2P', 'index': 0, 'type': 'tool_call_chunk'}]content='' additional_kwargs={'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': '{}', 'name': None}, 'type': None}]} response_metadata={} id='run-825b8400-30dd-4d02-ac24-47d0cc9c2cac' tool_calls=[{'name': '', 'args': {}, 'id': None, 'type': 'tool_call'}] tool_call_chunks=[{'name': None, 'args': '{}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]content='' additional_kwargs={} response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_bd83329f63'} id='run-825b8400-30dd-4d02-ac24-47d0cc9c2cac'

A `ValueError` will be raised if the caller tries to execute tools before the stream is exhausted.


```python
stream = chatbot.stream('Check my messages.')
for r in stream:
    print(r.content, end='', flush=True)
    break
try:
    stream.execute_tools()
except ValueError as e:
    print('Exception was caught!')
```

    Exception was caught!


## Toolkits and Tool Factories

Aside from providing a list of tools, you may also bind tools from toolkits and tool factories.

+ `ToolKit`: class with a `get_tools() -> list[BaseTool]` method. `ToolKit`s are part of the langchain interface, and the built-in tools often come as a subtype. Passed through the `toolkits: list[BaseToolkit]` argument.

+ ***Tool Factories***: functions that accept a chatbot as an argument and return tools. Useful when writing tools that interact with the original LLM because otherwise it would require partial initialization. Passed through the `tool_factories: ToolFactoryType` argument.

Note that these too may be provided at instantiation or at invokation.

#### `ToolKit` Example

In this example, I enable the built-in Langchain `FileManagementToolkit` toolkit to allow the chatbot to list, read, and write files.


```python
import tempfile
from langchain_community.agent_toolkits import FileManagementToolkit
with tempfile.TemporaryDirectory() as wd:
    file_tk = FileManagementToolkit(root_dir=str(wd))
    result = chatbot.invoke('List the files in this directory.', toolkits=[file_tk])
    print(result.tool_calls[0].tool_info_str())
```

    list_directory()


#### Tool Factory Examples
Now I create a tool factory that can be passed to the chatbot. This tool uses the chatbot reference to invoke the LLM with access to all of the same tools.


```python

def my_tool_factory(chatbot: simplechatbot.ChatBot) -> list[langchain_core.tools.Tool]:
    @langchain_core.tools.tool
    def story_generator(topic: str) -> str:
        '''Generate a story absed on a particular topic.'''
        result = chatbot.invoke(
            f'Generate a story about {topic}. Your response should only include the text of the story and make it short but engaging.',
        )
        return result.content

    return [story_generator]

result = chatbot.invoke('Generate a story about western cowboys.', tool_factories=[my_tool_factory])
tc_result = result.execute_tools()
tc_result['story_generator'].return_value
```




    'In the dusty town of Dry Gulch, the sun hung low in the sky, casting long shadows over the wooden saloons and weathered storefronts. The air was thick with the scent of leather and gunpowder, a reminder of the untamed land that surrounded the settlement. It was here that two rival cowboys, Jake “Iron” McGraw and Sam “Quickshot” Riley, were destined to cross paths.\n\nJake was known for his iron will and unmatched strength, while Sam earned his reputation as the fastest draw in the West. Their rivalry had begun years ago over a beautiful saloon owner, Clara, who had captured both their hearts with her fiery spirit. But Clara had made her choice, and it only intensified the flames of their competition.\n\nOne fateful afternoon, the townsfolk gathered in the dusty square, whispering tales of the upcoming showdown. With the sun dipping low, casting a golden hue over the landscape, Jake and Sam faced each other, eyes locked in a battle of wills. The tension was palpable, and the crowd held its breath.\n\n“Today, we settle this once and for all,” Jake growled, his hand hovering near his holster. Sam smirked, confidence radiating from him. “I’ve been waiting for this, Iron. Let’s see if you can keep up.”\n\nWith a single, thunderous clap of thunder, the duel began. Dust swirled around them as they moved like lightning, drawing their guns in a heartbeat. Shots rang out, echoing across the canyon. But in a twist of fate, both cowboys hesitated, their eyes catching Clara’s pleading gaze from the crowd.\n\nIn that moment, they realized the futility of their rivalry. The guns lowered, and silence fell. Clara rushed forward, tears in her eyes. “Enough! This isn’t what I wanted!”\n\nWith a shared look of understanding, Jake and Sam holstered their weapons. They turned to Clara, and together, they walked away from the battlefield, leaving behind the echoes of their past and forging a new path of friendship. The sun set over Dry Gulch, casting a warm glow as the three of them rode into the horizon, united against the world.'



## Conclusions
That is all! Now you know how to enable and disable tools that your LLM can use to do anything!
