# Tool Calling

`simplechatbot` empowers agents with the ability to produce arguments for arbitrary user functions instead of providing a text response to the user's prompt. Using this interface you can enable features such as web searching, email sending/checking, file browsing, image creation, or any other functionality that can be accessed through Python. The LLM will "decide" whether and which tools/functions should be executed based on a given prompt, so the key is to use tools with clear and concise instructions.

Under the hood, `ChatBot` instances maintain a collection of [langchain tools](https://python.langchain.com/docs/how_to/#tools) which can be extracted from toolkits or even factory methods that accept the chatbot itself as a parameter. Tools may also be added at the time of LLM execution to enable dynamic systems of available tools.

You can create your own [custom tools](https://python.langchain.com/docs/how_to/custom_tools/) or choose from [Langchain's built-in tools](https://python.langchain.com/docs/integrations/tools/). I will use `FileManagementToolkit` for demonstration purposes here.


```python
import sys
sys.path.append('../src/')

import simplechatbot
from simplechatbot.openai_agent import OpenAIAgent
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

agent = OpenAIAgent.new(
    model_name = 'gpt-4o-mini', 
    api_key=keychain['openai'],
    system_prompt=system_prompt,
    tools = [check_new_messages],
)
```

Now the LLM will have access to these tools. While the agent instance stores the LLM object in the `_model` attribute, you can use `model` to get the LLM with bound tools.


```python
agent.model
```




    RunnableBinding(bound=ChatOpenAI(client=<openai.resources.chat.completions.completions.Completions object at 0x10cd3c080>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x10cd3dd90>, root_client=<openai.OpenAI object at 0x10844b6e0>, root_async_client=<openai.AsyncOpenAI object at 0x10cd3c0e0>, model_name='gpt-4o-mini', model_kwargs={}, openai_api_key=SecretStr('**********')), kwargs={'tools': [{'type': 'function', 'function': {'name': 'check_new_messages', 'description': 'Check messages.', 'parameters': {'properties': {}, 'type': 'object'}}}]}, config={}, config_factories=[])



You can also use the method `get_model_with_tools` to get the tool-bound model with any additional tools. The `invoke`, `stream`, `chat`, and `chat_stream` methods all use this under-the hood so you can add any tools, toolkits, or tool factories to the model at invokation.


```python
agent.get_model_with_tools(tools=None)
```




    (RunnableBinding(bound=ChatOpenAI(client=<openai.resources.chat.completions.completions.Completions object at 0x10cd3c080>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x10cd3dd90>, root_client=<openai.OpenAI object at 0x10844b6e0>, root_async_client=<openai.AsyncOpenAI object at 0x10cd3c0e0>, model_name='gpt-4o-mini', model_kwargs={}, openai_api_key=SecretStr('**********')), kwargs={'tools': [{'type': 'function', 'function': {'name': 'check_new_messages', 'description': 'Check messages.', 'parameters': {'properties': {}, 'type': 'object'}}}]}, config={}, config_factories=[]),
     ToolLookup(tools={'check_new_messages': StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x1084516c0>)}))



Tools will be automatically used when we call any of the invoke or stream methods.

Notice that the LLM behaves normally if the user's prompts are unrelated to the tool.


```python
agent._invoke('Hello world!')
```




    ChatResult(content=Hello! How can I assist you today?, tool_calls=[])



If the LLM "decides" that the user needs to execute a tool, it returns a tool call as the response instead of returning content.


```python
result = agent._invoke('Check my messages.')
result
```




    ChatResult(content=, tool_calls=[ToolCallInfo(id='call_D0WyCNzn8Hg0qRxsui24aZ33', name='check_new_messages', type='tool_call', args={}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x1084516c0>))])



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

result = agent._invoke('Send a message to Bob saying "Hello!"', tools=[send_message])
result
```




    ChatResult(content=, tool_calls=[ToolCallInfo(id='call_CaZgudgmby8bQahiyL2PnP6J', name='send_message', type='tool_call', args={'recipient': 'Bob', 'text': 'Hello!'}, tool=StructuredTool(name='send_message', description='Send messages to others.', args_schema=<class 'langchain_core.utils.pydantic.send_message'>, func=<function send_message at 0x10cd45a80>))])



You can see that the LLM provided the `recipient` and `text` arguments which were passed to the function call information.


```python
result.tool_calls[0].tool_info_str()
```




    'send_message(recipient=Bob, text=Hello!)'



You can adjust behavior using the `tool_choice` argument in the chatbot constructor or at invokation. The value `'any'` means that a tool MUST be called, but all tools are candidates. The value `'auto'` (the default) allows the LLM to reply with normal content rather than a tool call, and you can also pass the name of a specific function as well.


```python
result = agent._invoke('Go to the store for me!', tool_choice='any')
result.tool_calls[0].tool_info_str()
```




    'check_new_messages()'



## Executing Tools
Tools allow the LLM to determine if and when to execute tools and also provides parameters for the tool call based on conversation history, but the user containing function is responsible for actually executing the tool with the arguments from the LLM.

Use the `execute_tools` method to actually execute the tool, which returns a mapping of tool names to `ToolCallResult` objects.


```python
result = agent._invoke('Check my messages.')
result.tool_calls[0].tool_info_str()
```




    'check_new_messages()'




```python
tr = result.execute_tools()
tr
```




    {'check_new_messages': ToolCallResult(info=ToolCallInfo(id='call_CPyijbPQpk91FIdl0B8NOLmI', name='check_new_messages', type='tool_call', args={}, tool=StructuredTool(name='check_new_messages', description='Check messages.', args_schema=<class 'langchain_core.utils.pydantic.check_new_messages'>, func=<function check_new_messages at 0x1084516c0>)), return_value='No new messages.')}



Get the return value from the tool through the `return_value` property.


```python
tr['check_new_messages'].return_value
```




    'No new messages.'



Extracting tool calls from a `StreamResult` is a little more complicated because the stream must be exhausted before executing tools. This happens because the tool call information replaces the text response, so the streamer is essentially receiving chunks of the function call information until exhaustion.

The calling function must handle both the streamed output and tool calls.


```python
stream = agent._stream('Check my messages.')
for r in stream:
    print(r, end='', flush=True)
if len(stream.tool_calls) > 0:
    stream.execute_tools()
```

    content='' additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_0I3oV4MeoavXZe8gyJygHkQZ', 'function': {'arguments': '', 'name': 'check_new_messages'}, 'type': 'function'}]} response_metadata={} id='run-029ed68d-e9ad-4a1f-9716-960302d7c64b' tool_calls=[{'name': 'check_new_messages', 'args': {}, 'id': 'call_0I3oV4MeoavXZe8gyJygHkQZ', 'type': 'tool_call'}] tool_call_chunks=[{'name': 'check_new_messages', 'args': '', 'id': 'call_0I3oV4MeoavXZe8gyJygHkQZ', 'index': 0, 'type': 'tool_call_chunk'}]content='' additional_kwargs={'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': '{}', 'name': None}, 'type': None}]} response_metadata={} id='run-029ed68d-e9ad-4a1f-9716-960302d7c64b' tool_calls=[{'name': '', 'args': {}, 'id': None, 'type': 'tool_call'}] tool_call_chunks=[{'name': None, 'args': '{}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]content='' additional_kwargs={} response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_06737a9306'} id='run-029ed68d-e9ad-4a1f-9716-960302d7c64b'

A `ValueError` will be raised if the caller tries to execute tools before the stream is exhausted.


```python
stream = agent.stream('Check my messages.')
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
    result = agent._invoke('List the files in this directory.', toolkits=[file_tk])
    print(result.tool_calls[0].tool_info_str())
```

    list_directory()


#### Tool Factory Examples
Now I create a tool factory that can be passed to the chatbot. This tool uses the chatbot reference to invoke the LLM with access to all of the same tools.


```python

def my_tool_factory(agent: simplechatbot.Agent) -> list[langchain_core.tools.Tool]:
    @langchain_core.tools.tool
    def story_generator(topic: str) -> str:
        '''Generate a story absed on a particular topic.'''
        result = agent._invoke(
            f'Generate a story about {topic}. Your response should only include the text of the story and make it short but engaging.',
        )
        return result.content

    return [story_generator]

result = agent._invoke('Generate a story about western cowboys.', tool_factories=[my_tool_factory])
tc_result = result.execute_tools()
tc_result['story_generator'].return_value
```




    'In the heart of the rugged Wild West, a small town called Dusty Springs thrived under the watchful eyes of the surrounding mesas. The sun blazed overhead, casting long shadows over the dusty streets where only a handful of souls dared to tread. Among them was Colt Harper, a seasoned cowboy known for his swift draw and uncanny ability to ride like the wind.\n\nOne sweltering afternoon, a mysterious stranger rode into town on a sleek black stallion. Cloaked in a dust-covered duster and a wide-brimmed hat that obscured his face, he dismounted at the saloon, causing the townsfolk to whisper in hushed tones. Colt, nursing a whiskey at the bar, felt a stirring in his gut; trouble had a way of finding him.\n\nThe stranger, known only as Jake, challenged Colt to a duel at high noon. Rumors swirled that Jake was looking for revenge against Colt’s brother, who had long since met his fate in a gunfight gone wrong. As the clock ticked towards noon, the townspeople gathered, tension crackling in the air like a summer storm.\n\nColt strode into the street, his boots kicking up dust as he faced Jake under the blazing sun. The world around them faded away, each heartbeat echoing in the silence. With a quick draw, Colt aimed and fired, the bullet whistling through the air and finding its mark. Jake fell, surprise blossoming on his face, and the whispering crowd gasped.\n\nAs the dust settled, Colt walked over to him. “This ain’t no way to settle a score,” he said, his voice steady. “Let’s put the past to rest. Life’s too short.” Jake nodded slowly, his anger replaced by a flicker of understanding.\n\nThe two men stood, embodying a spirit of camaraderie that echoed through the rugged landscape. Dusty Springs would remember that day not just for the gunfight, but for the moment a cowboy chose peace over vengeance. As the sun dipped below the horizon, Colt tipped his hat and rode off into the golden glow, a true gunslinger of the West.'



## Conclusions
That is all! Now you know how to enable and disable tools that your LLM can use to do anything!
