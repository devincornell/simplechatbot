
import datetime
from pydantic_ai import Agent, RunContext


myagent = Agent(
    model = 'openai:gpt-4o-mini',
    #deps_type=int,
    system_prompt = (
        "I'm a language model that can generate text based on the prompts you give me. "
        "I can write essays, poems, stories, and much more. "
        "I can also answer questions, summarize long texts, and translate languages. "
        "I can even write code in Python, JavaScript, and other programming languages. "
        "I can generate text in many different styles, such as formal, informal, academic, and creative. "
    )
)

@myagent.tool
async def get_date_and_time(ctx: RunContext[int]) -> str:
    '''Get the current date and time.'''
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(myagent.run_sync('What is the current date?'))
