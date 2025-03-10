import dataclasses
import typing

from .toolset import UknownToolError, ToolRaisedExceptionError
if typing.TYPE_CHECKING:
    from .agent import Agent
    from .chatresult import ChatResult
else:
    Agent = typing.TypeVar('Agent')
    ChatResult = typing.TypeVar('ChatResult')

@dataclasses.dataclass
class ChatBotUI:
    agent: Agent
    ignore_tool_exceptions: bool = False

    def start_interactive(self, 
        stream: bool = False,
        show_intro: bool = False,
        show_tools: bool = False,
    ) -> None:
        if show_intro:
            try:
                system_prompt = self.agent.history.first_system.content
                print('=============== System Message for this Chat ===================')
                print(system_prompt, '\n')
            except ValueError as e:
                pass
            try:
                tools = self.agent.toolset.render()
                print('\n=============== Tools for this Chat ===================')
                print(tools, '\n')
            except AttributeError:
                pass

        while True:
            if len(self.agent.history) and not self.agent.history.last.content.endswith('\n'):
                print()
            user_text = ''
            while not len(user_text.strip()):
                print('>> ', end='', flush=True)
                user_text = input()

            # process special commands
            if user_text in ('/exit', '/e'):
                break
            else:
                self._do_chat_call(user_text, stream, show_tools)

            #print(self.agent.history.get_buffer_string())

    def _do_chat_call(self, user_text: str|None, stream: bool, show_tools: bool) -> dict[str,ChatResult]:
        '''Handle a single chat. Recursive if tools are called.'''
        print(f'AI Response: ', end="", flush=True)
        if stream:
            result = self.agent.stream(user_text, add_to_history=True)
            for chunk in result:
                print(chunk.content, end="", flush=True)
        else:
            result = self.agent.chat(user_text, add_to_history=True)
            print(result.message.content)

        if self.ignore_tool_exceptions:
            try:
                tool_results = result.execute_tools()
            except UknownToolError as e:
                print(f'UNKOWN TOOL CALL: {e.tool_name}')

            except ToolRaisedExceptionError as e:
                print(f'TOOL RAISED EXCEPTION: {str(e.e)}\nCALL INFO: {e.tool_info}')
        else:
            tool_results = result.execute_tools()

        # if tools were called, do this recursively
        if len(tool_results):
            print('\n[Tool Results]')
            for tool_result in tool_results.values():
                print(f'{tool_result.info.tool_info_str()} -> {tool_result.return_value}')
            print('[END Tool Results]')

            # recursively call chat if tools were called
            self._do_chat_call(None, stream, show_tools)

    def start_streamlit(self, streamlit: typing.Any, show_intro: bool = False) -> None:
        user = lambda m: streamlit.chat_message("user").write(m)
        assistant = lambda m: streamlit.chat_message("assistant").write(m)
        if show_intro:
            try:
                system_prompt = self.agent.history.first_system.content
                assistant(f'=============== System Message for this Chat ===================\n\n{system_prompt}')
            except ValueError as e:
                pass
            try:
                tools = self.agent.toolset.render()
                assistant(f'=============== Tools for this Chat ===================\n\n{tools}')
            except AttributeError:
                pass

        if user_text := streamlit.text_input('>> '):
            #if len(self.agent.history) and not self.agent.history.last.content.endswith('\n'):
            #    print()
            #user_text = ''
            #while not len(user_text.strip()):
            #    print('>> ', end='', flush=True)
            #    user_text = input()
            # Display user input and save to message history.
            #self.agent.history.render_streamlit(streamlit)
            #print(len(self.agent.history))

            # print past history
            for msg in self.agent.history:
                streamlit.chat_message(msg.type).write(msg.content)

            user(user_text)
            #msgs.add_user_message(input)
            # Invoke chain to get reponse.
            #response = chain.invoke({'input': input})
            response = self.agent.chat(user_text)

            # Display AI assistant response and save to message history.
            assistant(str(response))
            #msgs.add_ai_message(response)
