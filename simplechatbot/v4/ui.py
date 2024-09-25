import dataclasses
import typing

from .toolset import UknownToolError, ToolRaisedExceptionError
if typing.TYPE_CHECKING:
    from .chatbot import ChatBot
else:
    ChatBot = typing.TypeVar('ChatBot')

@dataclasses.dataclass
class ChatBotUI:
    chatbot: ChatBot

    def start_interactive(self, 
        stream: bool = False,
        show_intro: bool = False,
        tool_verbose_callback: typing.Callable[[str],None]|None = None,
    ) -> None:
        if show_intro:
            try:
                system_prompt = self.chatbot.history.first_system.content
                print('=============== System Message for this Chat ===================')
                print(system_prompt, '\n')
            except ValueError as e:
                pass
            try:
                tools = self.chatbot.toolset.render()
                print('\n=============== Tools for this Chat ===================')
                print(tools, '\n')
            except AttributeError:
                pass

        while True:
            if len(self.chatbot.history) and not self.chatbot.history.last.content.endswith('\n'):
                print()
            user_text = ''
            while not len(user_text.strip()):
                print('>> ', end='', flush=True)
                user_text = input()

            # process special commands
            if user_text in ('/exit', '/e'):
                break
            else:
                try:
                    if stream:
                        for chunk in self.chatbot.chat_stream(user_text, tool_verbose_callback=tool_verbose_callback):
                            print(chunk, end="", flush=True)
                    else:
                        print(self.chatbot.chat(user_text, tool_verbose_callback=tool_verbose_callback))
                
                except UknownToolError as e:
                    print(f'UNKOWN TOOL CALL: {e.tool_name}')

                except ToolRaisedExceptionError as e:
                    print(f'TOOL RAISED EXCEPTION: {e.text}\n{e.tool_info}\n{e.e}')

    def start_streamlit(self, streamlit: typing.Any, show_intro: bool = False) -> None:
        user = lambda m: streamlit.chat_message("user").write(m)
        assistant = lambda m: streamlit.chat_message("assistant").write(m)
        if show_intro:
            try:
                system_prompt = self.chatbot.history.first_system.content
                assistant(f'=============== System Message for this Chat ===================\n\n{system_prompt}')
            except ValueError as e:
                pass
            try:
                tools = self.chatbot.toolset.render()
                assistant(f'=============== Tools for this Chat ===================\n\n{tools}')
            except AttributeError:
                pass

        if user_text := streamlit.text_input('>> '):
            #if len(self.chatbot.history) and not self.chatbot.history.last.content.endswith('\n'):
            #    print()
            #user_text = ''
            #while not len(user_text.strip()):
            #    print('>> ', end='', flush=True)
            #    user_text = input()
            # Display user input and save to message history.
            #self.chatbot.history.render_streamlit(streamlit)
            #print(len(self.chatbot.history))

            # print past istory
            for msg in self.chatbot.history:
                streamlit.chat_message(msg.type).write(msg.content)

            user(user_text)
            #msgs.add_user_message(input)
            # Invoke chain to get reponse.
            #response = chain.invoke({'input': input})
            response = self.chatbot.chat(user_text, tool_verbose_callback=assistant)

            # Display AI assistant response and save to message history.
            assistant(str(response))
            #msgs.add_ai_message(response)
