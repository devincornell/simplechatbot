


# Model-specific ChatBots

While `ChatBot` instances can be created from any Langchain Chat interface, we created some convenient superclasses that have varying levels of model-specific behavior.

Model-specific chatbots only differ from `ChatBot` in that they define static factory constructor methods, all named `new`. As each chat model needs to be installed separately, they must be accessed via separate imports.




---

``` python linenums="1"
import sys
sys.path.append('../src/')

import simplechatbot
```


---




I will use the keychain to manage API keys for OpenAI and Mistral.




---

``` python linenums="1"
keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')
```


---




Notice that we use a separate import statement to explicitly import the model-specific chatbots.




---

``` python linenums="1"
from simplechatbot.ollama_agent import OllamaAgent

agent = OllamaAgent.new(
    model_name = 'llama3.1', 
)
```


---





---

``` python linenums="1"
from simplechatbot.openai_agent import OpenAIAgent

agent = OpenAIAgent.new(
    model_name = 'gpt-4o-mini', 
    api_key=keychain['openai'],
)
```


---





---

``` python linenums="1"
from simplechatbot.mistral_agent import MistralAgent

agent = MistralAgent.new(
    model_name = 'mistral-large-latest', 
    api_key=keychain['mistral'],
)
```


---


 