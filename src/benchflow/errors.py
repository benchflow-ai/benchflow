from datetime import datetime, timezone
from typing import Any, Dict, Optional

class BenchflowError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
        }
    
    def __str__(self) -> str:
        return f"{self.code}: {self.message}, details={self.details})"

class BenchmarkNotFoundError(BenchflowError):
    def __init__(self, benchmark_name: str, details=None):
        template = f"Benchmark {benchmark_name} not found, 
                    benchmark_name should be all lowercase and follow the format: 'organization/benchmark_name'. 
                    Please check if the benchmark is available on https://www.benchflow.ai/dashboard/usages."
        super().__init__("BENCHMARK_NOT_FOUND", template, details)

class UnauthorizedError(BenchflowError):
    def __init__(self, api_key: str, details=None):
        api_key = api_key[:6] + "********" + api_key[-6:]
        template = f"Unauthorized access, please check your API key: {api_key} and try again."
        super().__init__("UNAUTHORIZED", template, details)

class UsageLimitExceededError(BenchflowError):
    def __init__(self, details=None):
        template = "Usage limit exceeded, please check your usage on https://www.benchflow.ai/dashboard/usages."
        super().__init__("USAGE_LIMIT_EXCEEDED", template, details)

class FunctionNotImplementedError(BenchflowError):
    def __init__(self, function_name: str, details=None):
        if function_name == "call_api":
            template = f"Function {function_name} not implemented, please check the BaseAgent class."
        else:
            template = f"Function {function_name} not implemented, please check the documentation on https://www.benchflow.ai/docs."
        super().__init__("FUNCTION_NOT_IMPLEMENTED", template, details)

class TaskInputFieldNotFoundError(BenchflowError):
    def __init__(
        self,
        missing_key: str,
        task_input_dict: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ):
        all_keys = list(task_input_dict.keys())
        template = (
            f"Task input field '{missing_key}' not found. "
            f"Available keys in the dictionary: {all_keys}."
        )
        super().__init__("TASK_INPUT_FIELD_NOT_FOUND", template, details)

class CallAPIExecutionError(BenchflowError):
    def __init__(self, details=None):
        template = "Error occurred while executing the API call"
        super().__init__("CALL_API_EXECUTION_ERROR", template, details)

class MissingParameterError(BenchflowError):
    def __init__(self, parameter_name: str, function_name: str, details=None):
        template = f"Missing parameter: {parameter_name} in function: {function_name}"
        super().__init__("MISSING_PARAMETER", template, details)

class InvalidArgumentsError(BenchflowError):
    def __init__(self, argument, details=None):
        message = f"Invalid arguments: {argument}"
        super().__init__("INVALID_ARGUMENTS", message, details)

class InvalidVersionError(BenchflowError):
    def __init__(self, version: str, details=None):
        template = f"Invalid version: {version}, please update your benchflow sdk to >= 0.1.13"
        super().__init__("INVALID_VERSION", template, details)











