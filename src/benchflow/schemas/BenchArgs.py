from typing import Any, Dict, List, Union

import yaml
from pydantic import BaseModel, Field, field_validator


from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
import yaml

class BenchArgs(BaseModel):
    # required arguments        
    required: List[str] = Field(default_factory=list)
    # optional arguments and their default values
    optional: Dict[str, Any] = Field(default_factory=dict)
    # specify choices for arguments, key is the argument name, value is the allowed values list
    choices: Dict[str, List[Any]] = Field(default_factory=dict)

    @field_validator('optional', mode='before')
    def merge_optional(cls, v):
        """
        If 'optional' is a list, merge each single key dictionary into a dictionary.
        """
        if isinstance(v, list):
            merged = {}
            for item in v:
                if isinstance(item, dict):
                    merged.update(item)
                else:
                    raise ValueError("Each item in 'optional' must be a dictionary")
            return merged
        elif isinstance(v, dict):
            return v
        else:
            raise ValueError("'optional' must be a dictionary or a list of dictionaries")

    def get_args(self, runtime_args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Merge runtime arguments with default values, validate required arguments, and validate values for arguments defined in choices.
        """
        runtime_args = runtime_args or {}
        args = {}

        # check required arguments
        for key in self.required:
            if key in runtime_args and runtime_args[key] is not None:
                args[key] = runtime_args[key]
            else:
                raise ValueError(f"Missing required argument: {key}")

        # process optional arguments: use runtime value if it exists, otherwise use default value
        for key, default in self.optional.items():
            args[key] = runtime_args.get(key, default)

        # validate arguments defined in choices
        for arg, allowed in self.choices.items():
            if arg in args and args[arg] not in allowed:
                raise ValueError(f"Invalid value for {arg}: {args[arg]}. Allowed values are: {allowed}")
        return args

    def __init__(self, config_source: Union[str, Dict[str, Any], None]):
        """
        The constructor can accept a YAML file path or a dictionary as the configuration source.
        """
        if isinstance(config_source, str):
            with open(config_source, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            data = config_source
        elif config_source is None:
            data = {"required": [], "optional": {}, "choices": {}}
        else:
            raise ValueError("config_source must be a YAML file path or a dictionary")
        super().__init__(**data)
