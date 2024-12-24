
def format_tool_text(tool_info: dict[str|dict]) -> str:
    '''Convenience method to format the function call to text.'''
    name = tool_info['name']
    args = tool_info['args']

    argtext = ', '.join(f"{k}={v}" for k,v in args.items())
    return f'{name}({argtext})'
