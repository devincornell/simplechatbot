# Structured Outputs
One of the most powerful features of LLMs is the ability to produce output which conforms to a pre-determined structure.

We define output structures using custom Pydantic types with attribute descriptions and validation logics. See more about using Pydantic types for structured outputs using langchain [here](https://python.langchain.com/docs/concepts/structured_outputs/). This package builds on that functionality by making it easy to request structured output at any point in an exchange.


```python
import pydantic

import sys
sys.path.append('../src/')

import simplechatbot
from simplechatbot.openai_agent import OpenAIAgent
```


```python

# optional: use this to grab keys from a json file rather than setting system variables
keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')

agent = OpenAIAgent.new('gpt-4o-mini', api_key=keychain['openai'])
print(agent)
```

    OpenAIAgent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={}))



```python
agent.history
```




    []




```python
agent.stream('Hello, how are you? My name is Devin. Don\'t forget it!').print_and_collect()
```

    Hello, Devin! I'm here to help you. What can I do for you today?




    ChatResult(content=Hello, Devin! I'm here to help you. What can I do for you today?, tool_calls=[])




```python
print(agent.history.get_buffer_string())
```

    Human: Hello, how are you? My name is Devin. Don't forget it!
    AI: Hello, Devin! I'm here to help you. What can I do for you today?



```python
class ResponseFormatter(pydantic.BaseModel):
    """Always use this tool to structure your response to the user."""
    answer: str = pydantic.Field(description="The answer to the user's question.")
    followup_question: str = pydantic.Field(description="A follow-up question the user could ask.")

agent.chat_structured('what is your favorite cat?', output_structure=ResponseFormatter)
```




    StructuredOutputResult(data=answer="I don't have personal preferences, but many people love the Maine Coon for their friendly nature and impressive size, or the Siamese for their vocal personalities and striking appearance. Do you have a favorite cat breed?" followup_question='What characteristics do you look for in a cat?')




```python
print(agent.history.get_buffer_string())
```

    Human: Hello, how are you? My name is Devin. Don't forget it!
    AI: Hello, Devin! I'm here to help you. What can I do for you today?
    Human: what is your favorite cat?
    AI: {"answer":"I don't have personal preferences, but many people love the Maine Coon for their friendly nature and impressive size, or the Siamese for their vocal personalities and striking appearance. Do you have a favorite cat breed?","followup_question":"What characteristics do you look for in a cat?"}



```python
agent.stream('I like that they are so cute!').print_and_collect()
```

    Cat cuteness is definitely a huge draw! Their playful antics, soft fur, and adorable faces can brighten anyone's day. Do you have a favorite cat or a pet of your own?




    ChatResult(content=Cat cuteness is definitely a huge draw! Their playful antics, soft fur, and adorable faces can brighten anyone's day. Do you have a favorite cat or a pet of your own?, tool_calls=[])




```python
print(agent.history.get_buffer_string())
```

    Human: Hello, how are you? My name is Devin. Don't forget it!
    AI: Hello, Devin! I'm here to help you. What can I do for you today?
    Human: what is your favorite cat?
    AI: {"answer":"I don't have personal preferences, but many people love the Maine Coon for their friendly nature and impressive size, or the Siamese for their vocal personalities and striking appearance. Do you have a favorite cat breed?","followup_question":"What characteristics do you look for in a cat?"}
    Human: I like that they are so cute!
    AI: Cat cuteness is definitely a huge draw! Their playful antics, soft fur, and adorable faces can brighten anyone's day. Do you have a favorite cat or a pet of your own?



```python
agent.chat_structured('what is your favorite cat?', output_structure=ResponseFormatter, add_to_history=False)
```




    StructuredOutputResult(data=answer="While I don't have personal favorites, many people adore the Ragdoll for its affectionate and calm temperament. They are known for their laid-back nature and striking blue eyes. What do you find cutest about cats?" followup_question='Have you ever thought about owning a cat?')




```python
print(agent.history.get_buffer_string())
```

    Human: Hello, how are you? My name is Devin. Don't forget it!
    AI: Hello, Devin! I'm here to help you. What can I do for you today?
    Human: what is your favorite cat?
    AI: {"answer":"I don't have personal preferences, but many people love the Maine Coon for their friendly nature and impressive size, or the Siamese for their vocal personalities and striking appearance. Do you have a favorite cat breed?","followup_question":"What characteristics do you look for in a cat?"}
    Human: I like that they are so cute!
    AI: Cat cuteness is definitely a huge draw! Their playful antics, soft fur, and adorable faces can brighten anyone's day. Do you have a favorite cat or a pet of your own?

