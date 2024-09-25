
import typing
import dataclasses
import json
import pathlib

@dataclasses.dataclass
class APIKeyChain:
    '''Manage API keys loaded from json files.'''
    fp: pathlib.Path
    keys: dict[str,str]

    @classmethod
    def from_json_file(cls, fname: str|pathlib.Path) -> dict[str,str]:
        '''Read a json file and extract API keys. Json file is just {"api_key_name" -> "api_key_value"}.'''
        fp = pathlib.Path(fname)
        with fp.open('r') as f:
            keys = json.load(f)
        
        return cls(
            fp = fp,
            keys = keys,
        )


    def __getitem__(self, key_name: str) -> str:
        '''Get the desired api key.'''
        return self.keys[key_name]

