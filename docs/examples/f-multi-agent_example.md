


# Multi-agent Examples

In this example I show an example of an agent that writes a story using a series of steps.

1. The user provides an idea for their story.
2. The `Outline Bot` generates an outline for the story including section titles and descriptions.
3. The first section content is created by the `Story Bot` from section title/description and the overall story description.
4. The `Summary Bot` creates a summary of the newly generated chapter.
5. The section summary is passed to the `Story Bot` along with section title/description to generate the next section.
6. All sections follow steps 3-5.
...
7. The sections are combined into a full story.

![Story bot design diagram](https://storage.googleapis.com/public_data_09324832787/story_bot_design.svg)




---

``` python linenums="1"
import pydantic
import typing

import sys
sys.path.append('../src/')

import simplechatbot
from simplechatbot.openai_agent import OpenAIAgent
```


---




First I create a new chatbot from OpenAI using the API stored in the keychain file.

I also create a new function `stream_it` that prints the result from the LLM as it is received and returns the full result of the LLM call.




---

``` python linenums="1"
keychain = simplechatbot.APIKeyChain.from_json_file('../keys.json')
base_agent = OpenAIAgent.new(
    model_name = 'gpt-4o-mini', 
    api_key=keychain['openai'],
)
```


---




Next we need to direct the LLM to create an outline for the story. I do this by creating a system prompt describing the task of creating an outline and a pydantic class to direct the LLM on how to structure its response. This is important because the output of the outline bot will be used for generating chapters.




---

``` python linenums="1"
class StoryOutline(pydantic.BaseModel):
    """Outline of the story."""

    story_topic: str = pydantic.Field(description="The topic of the story.")

    part1_title: str = pydantic.Field(description="Title of Part 1 of the story.")
    part1_description: str = pydantic.Field(description="Longer description of part 1.")

    part2_title: str = pydantic.Field(description="Title of Part 2 of the story.")
    part2_description: str = pydantic.Field(description="Longer description of part 2.")

    part3_title: str = pydantic.Field(description="Title of Part 3 of the story.")
    part3_description: str = pydantic.Field(description="Longer description of part 3.")

    def outline_str(self) -> str:
        return f'topic: {self.story_topic}\n\n{self.part1_title}: {self.part1_description}\n\n{self.part2_title}: {self.part2_description}\n\n{self.part3_title}: {self.part3_description}'


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

    def create_outline(self, story_description: str) -> StoryOutline:
        return self.agent.chat_structured(story_description, output_structure=StoryOutline).data

outline_bot = OutlineBot(base_agent)

q = f'Write an outline for a story about two friends who met when they were young and then lost touch. They meet again as adults and have to navigate their new relationship.'
outline: StoryOutline = outline_bot.create_outline(q)
print(outline.outline_str())
```



stdout:
 

    topic: Rekindling a Friendship
    
    The Innocent Bond: In a small town, two children, Mia and Jake, form a deep bond during summer vacations. They share adventures, secrets, and dreams, making a pact to always stay friends. However, as they grow older, Mia moves away, and they lose touch. The chapter ends with Mia looking back at old photos, reminiscing about the carefree days of their friendship and wondering how Jake is doing.
    
    A Chance Encounter: Years later, Mia returns to her hometown for a wedding. On the night of the event, she unexpectedly runs into Jake, who hasn't changed much but feels like a stranger. Their initial conversation is awkward, filled with nostalgia and uncertainty. But as the night unfolds, they find common ground in shared memories. They agree to grab coffee the next day to catch up, both feeling a mix of excitement and anxiety about seeing each other after so long.
    
    Rebuilding the Past: During coffee, Mia and Jake navigate the complexities of their adult lives, revealing new aspects of their personalities that have developed separately. They reminisce about the past but also confront the differences that have emerged over the years. Misunderstandings arise but are resolved through open communication, leading to a rekindling of their friendship. The chapter ends with both feeling hopeful but uncertain about what the future holds for their relationship.
    

 



---




Next we create a bot that will generate the actual chapter content based on information generated in the outline and a summary of the previous chapter (if one exists).




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
        story_topic: str,
        chapter_title: str,
        chapter_description: str,
        previous_chapter_summary: typing.Optional[str] = None,
    ) -> str:
        prompt = (
            f'General story topic: "{story_topic}"\n\n'
            f'Section title: "{chapter_title}"\n\n'
            f'Description: "{chapter_description}"\n\n'
            f'Previous chapter summary: "{previous_chapter_summary if previous_chapter_summary is not None else "No previous chapter - this is the first!"}"'
        )
        return self.agent.stream(prompt, add_to_history=False).print_and_collect().content

chapter_bot = ChapterBot(base_agent)
```


---




Now we actually create the chatper using the prompt.




---

``` python linenums="1"
chapter1 = chapter_bot.write_chapter(outline.story_topic, outline.part1_title, outline.part1_description)
```



stdout:
 

    The sun-drenched days of summer blended into a tapestry of laughter and adventure for Mia and Jake. Their small town, a canvas of vibrant colors and warm breezes, became the backdrop for their blossoming friendship. 
    
    They often met at the old oak tree, its branches spreading wide like welcoming arms, where they shared whispered secrets and dreams of the future. One day, they painstakingly crafted a treasure map from an old cereal box, declaring their mission to find hidden gems in the nearby woods. With their imaginations running wild, they transformed fallen twigs into swords, prowled like pirates, and crowned each other as guardians of their secret world. 
    
    Mia would sometimes pull out her worn notebook, filled with sketches of their adventures, while Jake would recount tales of daring knights and fearless dragons. They made a pact—an earnest promise to always stay friends—each believing their bond was unbreakable.
    
    Then, as summer slipped away and the cold winds of fate began to blow, Mia's family received the news. They were moving, a change that echoed with uncertainty and sadness. On the night before her departure, they sat under the stars, their hearts heavy with unspoken fears. Mia held Jake's hand tightly, sealing their pact with a desperate wish that distance would never come between them.
    
    Years passed like fluttering leaves in the autumn breeze. Mia moved to a place where everything felt foreign. New friends filled her days, but the whispers of that innocent bond lingered in the corners of her heart. One quiet evening, feeling nostalgic, she rummaged through a box of memories. 
    
    As she pulled out old photographs, the vibrant summer days unfolded before her eyes. There they were, sunlight dancing off their smiles, the world their playground. She paused at a picture where they both grinned widely, remnants of mud smeared on their cheeks from a day spent crafting makeshift capes in the backyard, imaginations soaring.
    
    Mia’s heart tightened as she wondered where Jake was now, if he recalled their pact, the whispers of their carefree laughter carried by the winds through the years. In that moment, she realized that though miles apart, the bond they forged during those innocent days still held a spark of hope, waiting to be rekindled.

 



---




Next I create a bot that will summarize a chapter. We will eventually feed this into the creation of chapter 2.




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

    def summarize(
        self,
        chapter_text: str,
    ) -> str:
        prompt = (
            f'Chapter text:\n\n{chapter_text}'
        )
        return self.agent.stream(prompt, add_to_history=False).print_and_collect().content

summary_bot = SummaryBot(base_agent)
```


---





---

``` python linenums="1"
ch1_summary = summary_bot.summarize(chapter1)
```



stdout:
 

    In this poignant chapter, we follow Mia and Jake, two friends who share a magical summer filled with adventures in their small town. Their friendship flourishes at the old oak tree, where they craft a treasure map and transform their imaginations into a world of pirates, knights, and dreams. They create a pact to always remain friends, believing their bond is unbreakable.
    
    However, their idyllic summer comes to an abrupt end when Mia's family announces they are moving away, leading to a heart-wrenching farewell under the starlit sky. Mia and Jake hold hands tightly, each silently wishing that distance would not diminish their friendship.
    
    As the years progress, Mia finds herself in a new, foreign place, making new friends but often reminiscing about her time with Jake. One evening, she discovers a box of memories, pulling out old photographs that remind her of their joyful days together. Reflecting on their shared adventures and the special bond they created, Mia wonders if Jake still remembers their pact. The chapter concludes with a sense of nostalgia and hope, highlighting that despite the miles between them, the spark of their friendship still glows, waiting for a chance to reignite.

 



---




Now we actually generate the second chapter using the outline and the previous chapter summary.




---

``` python linenums="1"
chapter2 = chapter_bot.write_chapter(outline.story_topic, outline.part2_title, outline.part2_description, ch1_summary)
```



stdout:
 

    Mia stepped out of the wedding venue, the soft glow of string lights illuminating the night as laughter and music floated through the air. She wrapped her shawl tighter around her shoulders, feeling a chill that had nothing to do with the evening breeze. It was a night of joy, yet a weight settled in her chest as she navigated the familiar streets of her hometown. 
    
    Suddenly, she spotted him across the courtyard – Jake, leaning casually against a pillar, his tall frame and easy smile still as comforting as she remembered. The years melted into the background as their eyes locked, and it felt as if time had rewound to that summer when everything was simple. But beneath the familiarity lay an awkwardness; he was a stranger now, the miles between them stretching wider than she’d anticipated. 
    
    “Jake?” Mia called out, her voice barely cutting through the thrum of the festivities. 
    
    “Mia!” he responded, his surprise mirrored in his bright eyes. He stepped forward, hands shoved in his pockets. “Wow, it’s been… what? Nearly a decade?”
    
    “Yeah, something like that.” She chuckled nervously, glancing away. “You look… well, the same.”
    
    He laughed, a sound that tugged at her heart, but it felt different this time — tinged with a hint of uncertainty. “And you still have that same spark,” he replied, though just as quickly as the words left his mouth, his expression faltered. “How’s life been?”
    
    They each began to share snippets of their lives, awkward pauses punctuating their conversation. The nostalgia laced their exchanges, memories of childhood adventures swirling around them. “Do you remember that treasure map we made?” Mia asked, a smile creeping onto her face.
    
    Jake’s face lit up. “Of course. We thought we were real pirates.” 
    
    The moment drew them closer, their shared laughter echoing in the night. As they reminisced, the tension melted away, and a flicker of connection sparked anew. Their hearts raced with an unspoken agreement: they wouldn’t let this chance pass by. 
    
    “How about we grab coffee tomorrow?” Mia suggested, holding her breath as she waited for his response.
    
    “Yeah, I’d like that,” Jake replied, his smile returning with a hint of warmth. 
    
    As they parted ways, a mix of excitement and anxiety washed over Mia. She watched him walk back into the crowd, feeling a renewed sense of hope. Perhaps this was the beginning of rekindling what they had lost, a chance to mend the spaces that distance had created.

 



---




Create a summary for chapter 2 now.




---

``` python linenums="1"
ch2_summary = summary_bot.summarize(chapter2)
```



stdout:
 

    In this chapter, Mia finds herself stepping out of a wedding venue, overwhelmed by a mix of joy and nostalgia. As she walks through her hometown, she unexpectedly encounters Jake, her childhood friend and first love, who she hasn't seen in nearly a decade. Their initial interaction is filled with awkwardness and familiarity, highlighting the emotional distance that time and circumstances have created between them. 
    
    They engage in small talk, reminiscing about their childhood adventures, such as creating a treasure map, which leads to laughter and a rekindling of their connection. Despite the years apart, they quickly find common ground, indicating an enduring bond beneath the surface. As Mia suggests meeting for coffee the next day, both characters sense a hopeful opportunity to explore what could be a revival of their relationship. The chapter concludes with Mia feeling a renewed sense of hope for the future as she watches Jake disappear back into the crowd, hinting at the possibility of mending the gaps that time had widened between them.

 



---




Use the summary of chapter 2 and outline information to create chapter 3.




---

``` python linenums="1"
chapter3 = chapter_bot.write_chapter(outline.story_topic, outline.part3_title, outline.part3_description, ch2_summary)
```



stdout:
 

    The coffee shop smelled of roasted beans and sweet pastries, a familiar scent that enveloped Mia as she entered, her heart racing with anticipation. She spotted Jake at a corner table, his back straight, absorbed in a steaming mug. Memories flooded her as she approached—of painting their childhood treehouse and sharing secrets under the stars.
    
    “Hey,” she said, her voice barely above a whisper. 
    
    He looked up, his expression brightening. “Hey, you made it.”
    
    They exchanged warm smiles, feeling the shock of re-encounter fade, replaced by a comfortable, yet cautious atmosphere. As they sipped their coffees, the conversation flowed from light-hearted banter to deeper confessions. Mia shared tales of her career struggles, and Jake spoke of unpredictable travel adventures. 
    
    “Remember the treasure map we made?” Mia chuckled, attempting to bridge any lingering gaps, her fingers tracing the rim of her cup.
    
    Jake laughed, a sound that reminded her of so many summers spent laughing together. “Yeah, and how we never found anything but those old coins?”
    
    A moment of silence lingered as they both realized how their paths have diverged since those days—her commitment to a steady job, his penchant for spontaneity. Tension arose as they stumbled into differences: Jake’s love for risk clashed with Mia’s yearning for stability.
    
    “This is nice, but…” Mia hesitated, searching for the right words, “could we talk about how life has changed us?”
    
    He nodded, looking serious for a moment. Misunderstandings emerged as they aired out old grievances—miscommunication that had haunted their friendship for years.
    
    “I thought you were mad at me for leaving,” Jake admitted, running a hand through his hair.
    
    “I was just scared of losing you,” Mia replied, her voice trembling slightly. They both took deep breaths, the tension easing as honesty laid the groundwork for rebuilding.
    
    With each revelation, walls crumbled, revealing the roots of their companionship buried beneath soil and time. They embraced vulnerability, each apology serving as both an acknowledgment of the past and a guide forward.
    
    They left the coffee shop with that same hopeful feeling—a cautious optimism decorating the edges of their hearts. As Mia turned to look at Jake one last time before parting ways, a sense of uncertainty lingered in the air. Would this be the beginning of something new, or merely a fleeting spark?

 



---




And now we can review the full text of the book.




---

``` python linenums="1"
story = f'''

Overview: {outline.story_topic}

== Chapter 1: {outline.part1_title} ==

{chapter1}


== Chapter 2: {outline.part2_title} ==

{chapter2}


== Chapter 3: {outline.part3_title} ==

{chapter3}

'''

import pprint
pprint.pprint(story)
```



stdout:
 

    ('\n'
     '\n'
     'Overview: Rekindling a Friendship\n'
     '\n'
     '== Chapter 1: The Innocent Bond ==\n'
     '\n'
     'The sun-drenched days of summer blended into a tapestry of laughter and '
     'adventure for Mia and Jake. Their small town, a canvas of vibrant colors and '
     'warm breezes, became the backdrop for their blossoming friendship. \n'
     '\n'
     'They often met at the old oak tree, its branches spreading wide like '
     'welcoming arms, where they shared whispered secrets and dreams of the '
     'future. One day, they painstakingly crafted a treasure map from an old '
     'cereal box, declaring their mission to find hidden gems in the nearby woods. '
     'With their imaginations running wild, they transformed fallen twigs into '
     'swords, prowled like pirates, and crowned each other as guardians of their '
     'secret world. \n'
     '\n'
     'Mia would sometimes pull out her worn notebook, filled with sketches of '
     'their adventures, while Jake would recount tales of daring knights and '
     'fearless dragons. They made a pact—an earnest promise to always stay '
     'friends—each believing their bond was unbreakable.\n'
     '\n'
     "Then, as summer slipped away and the cold winds of fate began to blow, Mia's "
     'family received the news. They were moving, a change that echoed with '
     'uncertainty and sadness. On the night before her departure, they sat under '
     "the stars, their hearts heavy with unspoken fears. Mia held Jake's hand "
     'tightly, sealing their pact with a desperate wish that distance would never '
     'come between them.\n'
     '\n'
     'Years passed like fluttering leaves in the autumn breeze. Mia moved to a '
     'place where everything felt foreign. New friends filled her days, but the '
     'whispers of that innocent bond lingered in the corners of her heart. One '
     'quiet evening, feeling nostalgic, she rummaged through a box of memories. \n'
     '\n'
     'As she pulled out old photographs, the vibrant summer days unfolded before '
     'her eyes. There they were, sunlight dancing off their smiles, the world '
     'their playground. She paused at a picture where they both grinned widely, '
     'remnants of mud smeared on their cheeks from a day spent crafting makeshift '
     'capes in the backyard, imaginations soaring.\n'
     '\n'
     'Mia’s heart tightened as she wondered where Jake was now, if he recalled '
     'their pact, the whispers of their carefree laughter carried by the winds '
     'through the years. In that moment, she realized that though miles apart, the '
     'bond they forged during those innocent days still held a spark of hope, '
     'waiting to be rekindled.\n'
     '\n'
     '\n'
     '== Chapter 2: A Chance Encounter ==\n'
     '\n'
     'Mia stepped out of the wedding venue, the soft glow of string lights '
     'illuminating the night as laughter and music floated through the air. She '
     'wrapped her shawl tighter around her shoulders, feeling a chill that had '
     'nothing to do with the evening breeze. It was a night of joy, yet a weight '
     'settled in her chest as she navigated the familiar streets of her '
     'hometown. \n'
     '\n'
     'Suddenly, she spotted him across the courtyard – Jake, leaning casually '
     'against a pillar, his tall frame and easy smile still as comforting as she '
     'remembered. The years melted into the background as their eyes locked, and '
     'it felt as if time had rewound to that summer when everything was simple. '
     'But beneath the familiarity lay an awkwardness; he was a stranger now, the '
     'miles between them stretching wider than she’d anticipated. \n'
     '\n'
     '“Jake?” Mia called out, her voice barely cutting through the thrum of the '
     'festivities. \n'
     '\n'
     '“Mia!” he responded, his surprise mirrored in his bright eyes. He stepped '
     'forward, hands shoved in his pockets. “Wow, it’s been… what? Nearly a '
     'decade?”\n'
     '\n'
     '“Yeah, something like that.” She chuckled nervously, glancing away. “You '
     'look… well, the same.”\n'
     '\n'
     'He laughed, a sound that tugged at her heart, but it felt different this '
     'time — tinged with a hint of uncertainty. “And you still have that same '
     'spark,” he replied, though just as quickly as the words left his mouth, his '
     'expression faltered. “How’s life been?”\n'
     '\n'
     'They each began to share snippets of their lives, awkward pauses punctuating '
     'their conversation. The nostalgia laced their exchanges, memories of '
     'childhood adventures swirling around them. “Do you remember that treasure '
     'map we made?” Mia asked, a smile creeping onto her face.\n'
     '\n'
     'Jake’s face lit up. “Of course. We thought we were real pirates.” \n'
     '\n'
     'The moment drew them closer, their shared laughter echoing in the night. As '
     'they reminisced, the tension melted away, and a flicker of connection '
     'sparked anew. Their hearts raced with an unspoken agreement: they wouldn’t '
     'let this chance pass by. \n'
     '\n'
     '“How about we grab coffee tomorrow?” Mia suggested, holding her breath as '
     'she waited for his response.\n'
     '\n'
     '“Yeah, I’d like that,” Jake replied, his smile returning with a hint of '
     'warmth. \n'
     '\n'
     'As they parted ways, a mix of excitement and anxiety washed over Mia. She '
     'watched him walk back into the crowd, feeling a renewed sense of hope. '
     'Perhaps this was the beginning of rekindling what they had lost, a chance to '
     'mend the spaces that distance had created.\n'
     '\n'
     '\n'
     '== Chapter 3: Rebuilding the Past ==\n'
     '\n'
     'The coffee shop smelled of roasted beans and sweet pastries, a familiar '
     'scent that enveloped Mia as she entered, her heart racing with anticipation. '
     'She spotted Jake at a corner table, his back straight, absorbed in a '
     'steaming mug. Memories flooded her as she approached—of painting their '
     'childhood treehouse and sharing secrets under the stars.\n'
     '\n'
     '“Hey,” she said, her voice barely above a whisper. \n'
     '\n'
     'He looked up, his expression brightening. “Hey, you made it.”\n'
     '\n'
     'They exchanged warm smiles, feeling the shock of re-encounter fade, replaced '
     'by a comfortable, yet cautious atmosphere. As they sipped their coffees, the '
     'conversation flowed from light-hearted banter to deeper confessions. Mia '
     'shared tales of her career struggles, and Jake spoke of unpredictable travel '
     'adventures. \n'
     '\n'
     '“Remember the treasure map we made?” Mia chuckled, attempting to bridge any '
     'lingering gaps, her fingers tracing the rim of her cup.\n'
     '\n'
     'Jake laughed, a sound that reminded her of so many summers spent laughing '
     'together. “Yeah, and how we never found anything but those old coins?”\n'
     '\n'
     'A moment of silence lingered as they both realized how their paths have '
     'diverged since those days—her commitment to a steady job, his penchant for '
     'spontaneity. Tension arose as they stumbled into differences: Jake’s love '
     'for risk clashed with Mia’s yearning for stability.\n'
     '\n'
     '“This is nice, but…” Mia hesitated, searching for the right words, “could we '
     'talk about how life has changed us?”\n'
     '\n'
     'He nodded, looking serious for a moment. Misunderstandings emerged as they '
     'aired out old grievances—miscommunication that had haunted their friendship '
     'for years.\n'
     '\n'
     '“I thought you were mad at me for leaving,” Jake admitted, running a hand '
     'through his hair.\n'
     '\n'
     '“I was just scared of losing you,” Mia replied, her voice trembling '
     'slightly. They both took deep breaths, the tension easing as honesty laid '
     'the groundwork for rebuilding.\n'
     '\n'
     'With each revelation, walls crumbled, revealing the roots of their '
     'companionship buried beneath soil and time. They embraced vulnerability, '
     'each apology serving as both an acknowledgment of the past and a guide '
     'forward.\n'
     '\n'
     'They left the coffee shop with that same hopeful feeling—a cautious optimism '
     'decorating the edges of their hearts. As Mia turned to look at Jake one last '
     'time before parting ways, a sense of uncertainty lingered in the air. Would '
     'this be the beginning of something new, or merely a fleeting spark?\n'
     '\n')
    

 



---


 