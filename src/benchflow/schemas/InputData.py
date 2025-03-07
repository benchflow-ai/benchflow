from typing import Any, Dict

from pydantic import BaseModel


class TaskStepInputs(BaseModel):
   env_info: Dict[str, Any] = None
