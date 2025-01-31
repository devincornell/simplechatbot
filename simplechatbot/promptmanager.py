
import typing
import dataclasses
import pathlib

import pydantic
import jinja2
import jinja2.meta

class PromptNotFound(Exception):
    '''Exception for when a prompt is not found.'''
    fpath: str | pathlib.Path
    fpath_txt: str | pathlib.Path

    @classmethod
    def from_path(cls, fpath: str, fpath_txt: str, fpath_md) -> typing.Self:
        o = cls(f'Prompt not found at path: {fpath} (also tried {fpath_txt} and {fpath_md})')
        o.fpath = fpath
        o.fpath_txt = fpath_txt
        return o

class TemplateVariableMismatch(Exception):
    '''Exception for when not all template variables are provided.'''
    provided: set[str]
    expected: set[str]

    @classmethod
    def from_expected(cls, provided: set[str], expected: set[str]) -> typing.Self:
        o = cls(f'Provided template variables do not match the input template. Missing: {expected-provided}; Extra: {provided-expected}.')
        o.provided = provided
        o.expected = expected
        return o
    
    @property
    def missing(self) -> set[str]:
        return self.expected - self.provided
    
    @property
    def extra(self) -> set[str]:
        return self.provided - self.expected



class PromptManager:
    '''Manage prompts for the chatbot.'''
    fpath: pathlib.Path
    def __init__(self, fpath: str | pathlib.Path):
        self.fpath = pathlib.Path(fpath)

    def get_prompt(self, path: str | pathlib.Path, strict: bool = True, template_vars: typing.Dict[str, typing.Any] | None = None) -> str:
        '''Get a prompt from the given path and render using template vars.'''
        full_path = self.fpath / path
        full_path_txt = full_path.with_suffix('.txt')
        full_path_md = full_path.with_suffix('.md')
        
        if full_path.exists():
            text = full_path.read_text()
        elif full_path_txt.exists():
            text = full_path_txt.read_text()
        elif full_path_md.exists():
            text = full_path_md.read_text()
        else:
            raise PromptNotFound.from_path(full_path, full_path_txt, full_path_md)

        return jinja_render(text, vars=template_vars or {}, strict=strict)

def jinja_render(
    input_text: str,
    vars: dict[str,typing.Any],
    strict: bool = True,
) -> str:
    '''Return the same document rendered as a jinja template.
    Args:
        input_text: the text to render.
        vars: the variables to substitute into the jinja template.
        strict: if True, raise an error if not all variables are provided.
    '''
    if strict:
        if (provided_vars := set(vars.keys())) != (expected_vars := set(jinja_get_variables(input_text))):
            raise TemplateVariableMismatch.from_expected(provided_vars, expected_vars)

    try:
        # NOTE: not sure which of these causes the exception
        template = text_to_jinja_template(input_text)
        rendered_text = template.render(vars)
    except jinja2.exceptions.TemplateSyntaxError as e:
        raise _add_line_number_to_exception_message(e)
    
    return rendered_text

def text_to_jinja_template(
    input_text: str,
    globals: dict[str, typing.Any] | None = None,
) -> jinja2.Template:
    '''Get a jinja template of the current document.'''
    env = jinja2.Environment()
    return env.from_string(
        source = input_text,
        globals = globals,
    )

def jinja_check_template_vars(input_text: str) -> None:
    '''Check if all jinja variables are provided.'''
    if jv := len(jinja_get_variables(input_text)):
        raise ValueError(f'Not all jinja template variables have been provided: {jv}')


def jinja_get_variables(input_text: str) -> list[str]:
    '''Get list of jinja variables to populate.
    Args:
        input_text: the text to look for variables in.
    '''
    env = jinja2.Environment()
    try:
        # NOTE: not sure which of these causes the exception
        parsed = env.parse(input_text)
        return list(jinja2.meta.find_undeclared_variables(parsed))
    except jinja2.exceptions.TemplateSyntaxError as e:
        raise _add_line_number_to_exception_message(e)

def _add_line_number_to_exception_message(
    e: jinja2.exceptions.TemplateSyntaxError
) -> jinja2.exceptions.TemplateSyntaxError:
    '''Add line number to error message of TemplateSyntaxError.'''
    if 'Missing end of comment tag' in e.message:
        addendum = ('This may be due to lack of whitespace around curly '
            'brackets in markdown classes. Use spacing such as "{ #id" '
            'when specifying an ID using pandoc\'s marking tool. It may '
            'otherwise be some other interaction between jinja and pandoc.'
        )
    else:
        addendum = ''

    return jinja2.exceptions.TemplateSyntaxError(
        message=f'{e.message} on line {e.lineno}. {addendum}',
        lineno=e.lineno,
        name=e.name,
        filename=e.filename,
    )