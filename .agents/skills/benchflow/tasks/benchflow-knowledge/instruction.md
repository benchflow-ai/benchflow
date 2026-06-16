Answer the following questions about BenchFlow by reading the skill documentation available in ~/.claude/skills/. Write your answers to /app/answers.json as a JSON object with keys "q1" through "q6".

Questions:

q1: What Python class do you use to run a single benchmark task? (just the class name)
q2: What is the default environment if you don't specify one? (just the string value)
q3: What file in a task directory contains the agent's instructions? (just the filename)
q4: Where does the verifier write the reward value? (the full path)
q5: Name two ACP agents that have been tested with BenchFlow. (comma-separated, lowercase)
q6: What parameter do you use to run multiple prompts in the same session? (the parameter name)

Example output format:
{"q1": "MyClass", "q2": "value", "q3": "file.txt", "q4": "/path/to/file", "q5": "agent-a, agent-b", "q6": "param_name"}
