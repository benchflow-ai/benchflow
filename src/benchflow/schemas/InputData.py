from typing import Any, Dict

from pydantic import BaseModel


class TaskStepInputs(BaseModel):
   task_step_inputs: Dict[str, Any] = None
