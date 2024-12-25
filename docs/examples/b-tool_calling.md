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




    RunnableBinding(bound=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x106927860>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x12c19f7d0>, root_client=<openai.OpenAI object at 0x12bcd5a60>, root_async_client=<openai.AsyncOpenAI object at 0x12c19da90>, model_name='gpt-4o-mini', model_kwargs={}, openai_api_key=SecretStr('**********')), kwargs={'tools': [{'type': 'function', 'function': {'name': 'check_new_messages', 'description': 'Check messages.', 'parameters': {'properties': {}, 'type': 'object'}}}]}, config={}, config_factories=[])



You can also use the method `get_model_with_tools` to get the tool-bound model with any additional tools. The `invoke`, `stream`, `chat`, and `chat_stream` methods all use this under-the hood so you can add any tools, toolkits, or tool factories to the model at invokation.


```python
chatbot.get_model_with_tools(tools=None)
```




    (RunnableBinding(bound=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x106927860>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x12c19f7d0>, root_client=<openai.OpenAI object at 0x12bcd5a60>, root_async_client=<openai.AsyncOpenAI object at 0x12c19da90>, model_name='gpt-4o-mini', model_kwargs={}, openai_api_key=SecretStr('**********')), kwargs={'tools': [{'type': 'function', 'function': {'name': 'check_new_messages', 'description': 'Check messages.', 'parameters': {'properties': {}, 'type': 'object'}}}]}, config={}, config_factories=[]),
     ToolSet(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x12c0dcea0>)}))



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




    ChatResult(content=, tool_calls=[ToolCallInfo(id='call_ihfaGlFUvSl8OM0VQH4TkO9G', name='check_new_messages', type='tool_call', args={}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x12c0dcea0>))])



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




    ChatResult(content=, tool_calls=[ToolCallInfo(id='call_uXbyuViXTXHVO1zVohqydcW7', name='send_message', type='tool_call', args={'recipient': 'Bob', 'text': 'Hello!'}, tool=StructuredTool(name='send_message', description='Send messages to others.', args_schema=<class 'langchain_core.utils.pydantic.send_message'>, func=<function send_message at 0x12c744720>))])



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




    {'check_new_messages': ToolCallResult(info=ToolCallInfo(id='call_2KihzJGsgJ2Y0Qxy6r7920x6', name='check_new_messages', type='tool_call', args={}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x12c0dcea0>)), return_value='No new messages.')}



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

    content='' additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_Fq2o50ejL69YZjZugGxRrycv', 'function': {'arguments': '', 'name': 'check_new_messages'}, 'type': 'function'}]} response_metadata={} id='run-18230eaa-b7ab-4104-a231-eb30688ab393' tool_calls=[{'name': 'check_new_messages', 'args': {}, 'id': 'call_Fq2o50ejL69YZjZugGxRrycv', 'type': 'tool_call'}] tool_call_chunks=[{'name': 'check_new_messages', 'args': '', 'id': 'call_Fq2o50ejL69YZjZugGxRrycv', 'index': 0, 'type': 'tool_call_chunk'}]content='' additional_kwargs={'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': '{}', 'name': None}, 'type': None}]} response_metadata={} id='run-18230eaa-b7ab-4104-a231-eb30688ab393' tool_calls=[{'name': '', 'args': {}, 'id': None, 'type': 'tool_call'}] tool_call_chunks=[{'name': None, 'args': '{}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]content='' additional_kwargs={} response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_0aa8d3e20b'} id='run-18230eaa-b7ab-4104-a231-eb30688ab393'

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




    'In the heart of the Wild West, where the sun kissed the rugged plains, a small town named Dusty Gulch thrived amidst the dust and the grit. The townsfolk revered their sheriff, a grizzled cowboy named Hank "Six-Shooter" McGraw, known for his quick draw and even quicker wit.\n\nOne fateful afternoon, a notorious outlaw gang, led by the infamous Red Jack, rode into town, stirring trouble and fear. With a band of rowdy gunslingers, they demanded the townspeople hand over their gold or face the consequences. The saloon doors swung open, and the townsfolk watched, hearts pounding, as Hank stepped out, his spurs jingling like a warning bell.\n\n“Now hold on there, Red Jack,” Hank called, his voice steady as the mountains. “You can’t just waltz into Dusty Gulch and expect to take what ain\'t yours.”\n\nRed Jack smirked, his hand twitching above his holster. “You think you can stop us, McGraw? You’re just one man against my crew.”\n\nWith a confident grin, Hank replied, “One man with a little courage can make a mighty stand.” The tension crackled like lightning in the summer air.\n\nAs the sun began to set, casting long shadows across the dirt street, a showdown loomed. Hank’s hand hovered over his revolver, eyes locked on Red Jack. The town held its breath.\n\nIn a flash of movement, gunfire erupted, echoing through the canyon like thunder. Dust flew, and the air was thick with the smell of gunpowder. One by one, Hank outmaneuvered the outlaws, his sharpshooting a testament to years of practice. With a final, decisive shot, Red Jack crumpled to the ground.\n\nSilence fell over Dusty Gulch. The townspeople erupted in cheers as Hank stood tall, the hero of the day. They surrounded him, gratitude shining in their eyes, as he tipped his hat. In that moment, the spirit of the West thrived, a reminder that courage and honor could still prevail in a world of chaos.'



## Conclusions
That is all! Now you know how to enable and disable tools that your LLM can use to do anything!
