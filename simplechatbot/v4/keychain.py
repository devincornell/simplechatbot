
import typing
import json
import pathlib

class APIKeyChain(typing.Dict[str,str]):
    '''Subclass dict with sfcm from_json_file(), so works as regular dict.
    Description: Manages API keys loaded from json files.
    '''

    @classmethod
    def from_json_file(cls, fname: str|pathlib.Path) -> dict[str,str]:
        '''Read a json file and extract API keys. Json file is just {"api_key_name" -> "api_key_value"}.'''
        fp = pathlib.Path(fname)
        with fp.open('r') as f:
            keys = json.load(f)
        
        return cls(keys)

