# Model-specific ChatBots

While `ChatBot` instances can be created from any Langchain Chat interface, we created some convenient superclasses that have varying levels of model-specific behavior.

Model-specific chatbots only differ from `ChatBot` in that they define static factory constructor methods, all named `new`. As each chat model needs to be installed separately, they must be accessed via separate imports.


```python
import sys
sys.path.append('..')

import simplechatbot
```

I will use the keychain to manage API keys for OpenAI and Mistral.


```python
keychain = simplechatbot.devin.APIKeyChain.from_json_file('../keys.json')
```

Notice that we use a separate import statement to explicitly import the model-specific chatbots.


```python
from simplechatbot.devin.ollama import OllamaChatBot

chatbot = OllamaChatBot.new(
    model_name = 'llama3.1', 
)
```


```python
from simplechatbot.devin.openai import OpenAIChatBot

chatbot = OpenAIChatBot.new(
    model_name = 'gpt-4o-mini', 
    api_key=keychain['openai'],
)
```


```python
from simplechatbot.devin.mistral import MistralChatBot

chatbot = MistralChatBot.new(
    model_name = 'mistral-large-latest', 
    api_key=keychain['mistral'],
)
```
