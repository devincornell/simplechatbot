


# Structured Outputs
One of the most powerful features of LLMs is the ability to produce outputs which conform to a pre-determined structure. There are several instances in which this feature is critical.

+ Output structures may enforce "reasoning" or systematic approaches to generating responses.
+ The output must be passed to an application or interface that is not simply text-based.

We define output structures using custom Pydantic types with attribute descriptions and validation logics. See more about using Pydantic types for structured outputs using langchain [here](https://python.langchain.com/docs/concepts/structured_outputs/). This package builds on that functionality by making it easy to request structured output at any point in an exchange.




---

``` python linenums="1"
import pydantic

import sys
sys.path.append('../src/')

import simplechatbot
from simplechatbot.openai_agent import OpenAIAgent
```


---




Create a new basic agent with no system prompt and no tools.




---

``` python linenums="1"
# optional: use this to grab keys from a json file rather than setting system variables
keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')

agent = OpenAIAgent.new('gpt-4o-mini', api_key=keychain['openai'])
print(agent)
```



stdout:
 

    OpenAIAgent(model_type=ChatOpenAI, model_name="gpt-4o-mini", tools=ToolLookup(tools={}))
    

 



---




Note that this is a normal agent that can be conversed with.




---

``` python linenums="1"
agent.stream('Hello, how are you? My name is Devin. Don\'t forget it!').print_and_collect()
```



stdout:
 

    Hello, Devin! I'm doing well, thank you. How can I assist you today?

 





text:

    ChatResult(content=Hello, Devin! I'm doing well, thank you. How can I assist you today?, tool_calls=[])
 


 


 



---




## Pydantic Types and `chat_structured`

The `chat_structured` method allows you to provide an output structure that the LLM response will be constrained to. The example below forces the LLM to answer the question and even come up with a follow-up question.




---

``` python linenums="1"
class ResponseFormatter(pydantic.BaseModel):
    """Always use this tool to structure your response to the user."""
    answer: str = pydantic.Field(description="The answer to the user's question.")
    followup_question: str = pydantic.Field(description="A follow-up question the user could ask.")

sresult = agent.chat_structured('what is your favorite cat?', output_structure=ResponseFormatter)
sresult
```




text:

    StructuredOutputResult(data=answer="As an AI, I don't have personal preferences or feelings, but popular cat breeds that many people love include the Maine Coon for their friendly nature and size, the Siamese for their vocal personality, and the Ragdoll for their affectionate demeanor. What about you, Devin? Do you have a favorite cat?" followup_question='What qualities do you look for in a favorite cat?')
 


 


 



---




As you can see, the `chat_structured` method returns a `StructuredOutputResult` instance, which has a `data` attribute which actually stores the Pydantic type instance.




---

``` python linenums="1"
sresult.data
```




text:

    ResponseFormatter(answer="As an AI, I don't have personal preferences or feelings, but popular cat breeds that many people love include the Maine Coon for their friendly nature and size, the Siamese for their vocal personality, and the Ragdoll for their affectionate demeanor. What about you, Devin? Do you have a favorite cat?", followup_question='What qualities do you look for in a favorite cat?')
 


 


 



---




Access the attributes of the response through this instance.




---

``` python linenums="1"
sresult.data.answer
```




text:

    "As an AI, I don't have personal preferences or feelings, but popular cat breeds that many people love include the Maine Coon for their friendly nature and size, the Siamese for their vocal personality, and the Ragdoll for their affectionate demeanor. What about you, Devin? Do you have a favorite cat?"
 


 


 



---





---

``` python linenums="1"
sresult.data.followup_question
```




text:

    'What qualities do you look for in a favorite cat?'
 


 


 



---




You can see this object in json format using the `as_json` method.




---

``` python linenums="1"
print(sresult.as_json(indent=2))
```



stdout:
 

    {
      "answer": "As an AI, I don't have personal preferences or feelings, but popular cat breeds that many people love include the Maine Coon for their friendly nature and size, the Siamese for their vocal personality, and the Ragdoll for their affectionate demeanor. What about you, Devin? Do you have a favorite cat?",
      "followup_question": "What qualities do you look for in a favorite cat?"
    }
    

 



---




You can see that the response was included in the conversation history in json format. Fortunately, modern LLMs can easily parse json data structures to keep track of conversation progress.




---

``` python linenums="1"
print(agent.history.get_buffer_string())
```



stdout:
 

    Human: Hello, how are you? My name is Devin. Don't forget it!
    AI: Hello, Devin! I'm doing well, thank you. How can I assist you today?
    Human: what is your favorite cat?
    AI: {"answer":"As an AI, I don't have personal preferences or feelings, but popular cat breeds that many people love include the Maine Coon for their friendly nature and size, the Siamese for their vocal personality, and the Ragdoll for their affectionate demeanor. What about you, Devin? Do you have a favorite cat?","followup_question":"What qualities do you look for in a favorite cat?"}
    

 



---




Asking it again by passing `new_message=None` shows that it will simply ask the questions in plain text format, so it will re-use the structured response provided in the previous message.




---

``` python linenums="1"
agent.stream(None).print_and_collect()
```



stdout:
 

    As an AI, I don't have personal preferences or feelings, but popular cat breeds that many people love include the Maine Coon for their friendly nature and size, the Siamese for their vocal personality, and the Ragdoll for their affectionate demeanor. What about you, Devin? Do you have a favorite cat?

 





text:

    ChatResult(content=As an AI, I don't have personal preferences or feelings, but popular cat breeds that many people love include the Maine Coon for their friendly nature and size, the Siamese for their vocal personality, and the Ragdoll for their affectionate demeanor. What about you, Devin? Do you have a favorite cat?, tool_calls=[])
 


 


 



---




## Complex Structures and Response Order

The most powerful aspect of structured responses is that they force the LLM to provide parts of the full response separately and in-sequence. Carefully designed output structures can lead to better and more complete responses.

In the following questions, we ask the LLM to provide a response to the question "Why are cheetahs so fast?" with different output structures.




---

``` python linenums="1"
class ReasonedResponse(pydantic.BaseModel):
    """Always use this tool to structure your response to the user."""
    answer: str = pydantic.Field(description="The answer to the user's question.")
    reasons: list[str] = pydantic.Field(description="Reasons for the answer.")

sresult = agent.chat_structured('Why are cheetahs so fast?', output_structure=ReasonedResponse, add_to_history=False)
print(sresult.as_json(indent=2))
```



stdout:
 

    {
      "answer": "Cheetahs are fast due to their unique physical adaptations and evolutionary traits.",
      "reasons": [
        "Cheetahs have a lightweight body structure, which reduces drag when running.",
        "They possess long, powerful legs that enable rapid acceleration and speed.",
        "Their flexible spine allows for an extended stride length, increasing their speed.",
        "Cheetahs have large nasal passages and lungs that facilitate increased oxygen intake during high-speed chases."
      ]
    }
    

 



---





---

``` python linenums="1"
class ReasonedResponse2(pydantic.BaseModel):
    """This is a well-reasoned response."""
    reasons: list[str] = pydantic.Field(description="Reasons for the answer.")
    answer: str = pydantic.Field(description="The answer to the user's question.")

sresult = agent.chat_structured('Why are cheetahs so fast?', output_structure=ReasonedResponse2, add_to_history=False)
print(sresult.as_json(indent=2))
```



stdout:
 

    {
      "reasons": [
        "Cheetahs have a lightweight body structure and long legs designed for speed.",
        "Their flexible spine allows for an extended stride length while running.",
        "Muscle composition in cheetahs is optimized for quick bursts of speed, containing a high percentage of fast-twitch muscle fibers.",
        "They have large nasal passages for increased oxygen intake and a specialized respiratory system that facilitates rapid breathing during sprints."
      ],
      "answer": "Cheetahs are so fast due to their lightweight body, long leg structure, flexible spine, and muscle composition optimized for speed."
    }
    

 



---




You may also create nested response structures. In this case, we further improve reasoning abilities by requiring the LLM to self-rate the quality of its own responses, leading to a final answer which places special emphasis on these reasons.




---

``` python linenums="1"
class SingleReason(pydantic.BaseModel):
    """This is a single reason and a self-assessment of the quality of the reason."""
    reason: str = pydantic.Field(description="A description of the reason.")
    quality: int = pydantic.Field(description="Quality score represented by an integer between 1 and 10.")

class ReasonedResponse3(pydantic.BaseModel):
    """This is a well-reasoned response."""
    reasons: list[SingleReason] = pydantic.Field(description="Reasons for the answer.")
    answer: str = pydantic.Field(description="The answer to the user's question, based on the given reasons and emphsaizing the highest quality reasons.")

sresult = agent.chat_structured('Why are cheetahs so fast?', output_structure=ReasonedResponse3, add_to_history=False)
print(sresult.as_json(indent=2))
```



stdout:
 

    {
      "reasons": [
        {
          "reason": "Cheetahs have a lightweight body structure that reduces drag and allows for quicker acceleration.",
          "quality": 9
        },
        {
          "reason": "Their leg muscles are highly specialized for sprinting, providing powerful and rapid movement.",
          "quality": 8
        },
        {
          "reason": "Cheetahs possess large nostrils that allow for increased oxygen intake during a sprint, supporting their high-speed chases.",
          "quality": 7
        },
        {
          "reason": "Their flexible spine enables a longer stride length while running, enhancing speed.",
          "quality": 8
        }
      ],
      "answer": "Cheetahs are so fast due to their lightweight body structure, specialized leg muscles, large nostrils for oxygen intake, and a flexible spine that allows for longer strides."
    }
    

 



---




Different approaches for output structures and ordering can lead to vastly different results, so, as with most Generative AI applications, experimentation is essential!

 