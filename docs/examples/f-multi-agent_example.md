


# Multi-agent Examples

In this example I show an example of an agent that writes a story using a series of steps.

1. The user provides an idea for their story.
2. `OutlineBot` generates an outline for the story including a topic, narrative arc, title, and individual chapter narratives, events, and titles.
3. `StoryBot` creates the first chapter based on the chatper and overall story outines.
4. `SummaryBot` creates a summary of the newly generated chapter.
5. The chapter summary is passed to `StoryBot` along with section the outline to generate the next chapter.
6. All chapter follow steps 3-5.
...
7. The chapter are combined into a full story.

![Story bot design diagram](https://storage.googleapis.com/public_data_09324832787/story_bot_design.svg)




---

``` python linenums="1"
import pydantic
import typing
import pathlib

import sys
sys.path.append('../src/')

import simplechatbot
from simplechatbot.openai_agent import OpenAIAgent
from simplechatbot.ollama_agent import OllamaAgent
```


---




First I create a new agent from OpenAI using the API stored in the keychain file. Most importantly, the agent maintains a LangChain chat model so we can use the `base_agent` to create new application-specific agents.




---

``` python linenums="1"
if True:
    keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')
    base_agent = OpenAIAgent.new(
        model_name = 'gpt-4o-mini', 
        api_key=keychain['openai'],
    )
else:
    base_agent = OllamaAgent.new(
        model_name = 'llama3.1', 
    )
```


---




Next we create the `StoryOutline` pydantic type to describe the structure of the output. Notice that the `StoryOutline` has a variable number of chapters, each with its own narrative, events, and title. Also note that the attribute order for `StoryOutline` follows the flow of topic, narrative, chapters, and then finally title. The values will be generated in that order, so we choose an ordering that most effectively can capture the process of producing creative stories.

Also essential to this agent is the system prompt, which can be adjusted given feedback about outline qualities.




---

``` python linenums="1"
class ChapterOutline(pydantic.BaseModel):
    """Outline of a chapter in a story."""
    narrative_arc: str = pydantic.Field(description="The narrative arc of the story.")
    events: typing.List[str] = pydantic.Field(description="List of the most important events that happen in the chapter.")
    title: str = pydantic.Field(description="Title of the chapter.")


class StoryOutline(pydantic.BaseModel):
    """Outline of the story."""
    topic: str = pydantic.Field(description="The main topic of the story.")
    narrative_arc: str = pydantic.Field(description="The narrative arc of the story.")
    chapters: list[ChapterOutline] = pydantic.Field(description="List of chapters in the story.")
    title: str = pydantic.Field(description="Title of the story.")

class OutlineBot:
    system_prompt = (
        "The user will provide you a description of a story, and you must create a chapter outline with titles and brief descriptions. "
        "Each section should contain a title and a brief description of what happens in that section. "
        "The sections should all be part of a single narrative ark, but each section should have a complete beginning, middle, and end. "
    )
    def __init__(self, base_agent: simplechatbot.Agent):
        self.agent = base_agent.new_agent_from_model(
            system_prompt=self.system_prompt,
        )

    def create_outline(self, story_description: str) -> simplechatbot.StructuredOutputResult:
        return self.agent.chat_structured(story_description, output_structure=StoryOutline)

outline_bot = OutlineBot(base_agent)
outline_bot
```




text:

    <__main__.OutlineBot at 0x109b68770>
 


 


 



---




Next we create `ChapterBot`, which will generate the actual chapter content based on information generated in the outline and a summary of the previous chapter (if one exists).




---

``` python linenums="1"
class ChapterBot:
    system_prompt = (
        "You are designed to write a single chapter of a larger story based on the following information: \n\n"
        "+ Overall story topic: The topic of the full story.\n"
        "+ Chapter title: The title of the section you are writing.\n"
        "+ Chapter description: A longer description of what happens in the section.\n"
        "+ (optional) previous chapter summary: A summary of the previous chapter.\n"
        "Your responses should only include text that is part of the story. Do not include the chapter title \n"
        "or any other information that is not part of the story itself.\n"
        "The story chapter should be super short, so keep that in mind!"
    )

    def __init__(self, base_agent: simplechatbot.Agent):
        self.agent = base_agent.new_agent_from_model(
            system_prompt=self.system_prompt,
        )

    def write_chapter(
        self,
        story_outline: StoryOutline,
        chapter_outline: ChapterOutline,
        prev_chapter_summary: typing.Optional[str] = None,
    ) -> str:
        prompt = (
            f'General story topic: "{story_outline.topic}"\n\n'
            f'Chapter title: "{chapter_outline.title}"\n\n'
            f'Chapter narrative arc: "{chapter_outline.narrative_arc}"\n\n'
            f'Previous chapter summary: "{prev_chapter_summary if prev_chapter_summary is not None else "No previous chapter - this is the first!"}"'
        )
        return self.agent.stream(prompt, add_to_history=False).progress_and_collect().content

chapter_bot = ChapterBot(base_agent)
```


---




Finally we create `SummaryBot`, which has the simple task of generating a summary of a given chapter.




---

``` python linenums="1"
class SummaryBot:
    system_prompt = (
        'You need to create a summary of the story chapter provided to you by the user. '
        'The summary should include names of relevant characters and capture the story arc of the chapter. '
    )

    def __init__(self, base_agent: simplechatbot.Agent):
        self.agent = base_agent.new_agent_from_model(
            system_prompt=self.system_prompt,
        )

    def summarize(self, chapter_text: str) -> str:
        prompt = (
            f'Chapter text:\n\n{chapter_text}'
        )
        return self.agent.stream(prompt, add_to_history=False).progress_and_collect().content

summary_bot = SummaryBot(base_agent)
```


---




Now we can create a `Story` type to tie it all together: it will contain the overal narrative arc as well as individual chapter outline, text, and summaries. The `generate_from_topic` method accepts a topic and from there will generate an outline and sequentially generate chapters and chapter summaries of the story.




---

``` python linenums="1"
import tqdm

class Chapter(pydantic.BaseModel):
    i: int
    outline: ChapterOutline
    text: str
    summary: str

class Story(pydantic.BaseModel):
    outline: StoryOutline
    chapters: typing.List[Chapter] = pydantic.Field(default_factory=list)

    @classmethod
    def generate_from_topic(
        cls,
        topic: str,
        outline_bot: OutlineBot,
        chapter_bot: ChapterBot,
        summary_bot: SummaryBot,
    ) -> typing.Self:
        outline_result = outline_bot.create_outline(topic)
        outline: StoryOutline = outline_result.data

        chapters = list()
        prev_summary = None
        for i, chapter in enumerate(tqdm.tqdm(outline.chapters, ncols=80)):
            chapter_text = chapter_bot.write_chapter(
                story_outline=outline, 
                chapter_outline=chapter,
                prev_chapter_summary=prev_summary,
            )
            summary = summary_bot.summarize(chapter_text)
            prev_summary = summary
            chapters.append(Chapter(i=i, outline=chapter, text=chapter_text, summary=summary))
        return cls(outline=outline, chapters=chapters)
    
    def add_chapter(
        self, 
        outline: ChapterOutline,
        text: str,
        summary: str,
    ) -> None:
        self.chapters.append(Chapter(outline=outline, text=text, summary=summary))
        
```


---




Finally, we define a topic and generate a new story!




---

``` python linenums="1"
q = (
    f'The story should be about two friends who met when they were young and then lost touch.'
    'They meet again as adults and have to navigate their new relationship. '
)

story = Story.generate_from_topic(
    topic=q,
    outline_bot=outline_bot,
    chapter_bot=chapter_bot,
    summary_bot=summary_bot,
)
```



stderr:
 

    342it [00:05, 63.03it/s]
    188it [00:02, 81.52it/s]
    257it [00:03, 72.37it/s]
    176it [00:02, 78.58it/s]
    439it [00:05, 79.36it/s]
    128it [00:01, 82.71it/s] 
    355it [00:04, 82.32it/s]
    185it [00:04, 38.40it/s]
    457it [00:05, 80.11it/s]
    202it [00:02, 78.42it/s]
    355it [00:04, 81.00it/s]
    163it [00:01, 85.35it/s] 
    6it [00:44,  7.39s/it]
    

 



---





---

``` python linenums="1"
story.outline.title
```




text:

    'Bridges to the Past'
 


 


 



---





---

``` python linenums="1"
print(story.outline.narrative_arc)
```



stdout:
 

    Two childhood friends, separated by life changes, reunite as adults and must grapple with their evolution while rekindling their bond.
    

 



---





---

``` python linenums="1"
dest = pathlib.Path('story_results/')
dest.mkdir(parents=True, exist_ok=True)

overview = f'''{story.outline.title}
{len(story.outline.chapters)} Chapters

Topic: {story.outline.topic}

--------------------------------
Narrative:
{story.outline.narrative_arc}
--------------------------------
Chapters:

'''
overview += '\n\n'.join([f'{c.title}\n{c.narrative_arc}' for c in story.outline.chapters])
with dest.joinpath('overview.txt').open('w') as f:
    f.write(overview)

for chapter in story.chapters:
    events = '\n'.join([f'\t{i+1}. {e}' for i, e in enumerate(chapter.outline.events)])
    with dest.joinpath(f'{chapter.i}. {chapter.outline.title}.txt').open('w') as f:
        f.write(f'{chapter.outline.title}\n\nNarrative: {chapter.outline.narrative_arc}\n\nEvents:\n{events}\n\n------------------\nSummary:\n{chapter.summary}\n\n\n------------------\n{chapter.text}\n\n')
```


---


 