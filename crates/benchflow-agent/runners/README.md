# Runner scripts for SDK agents

Some agents are a framework rather than a command-line program. For each, BenchFlow ships a runner script here that builds the agent from the framework, runs it once on the instruction, and reports what it does as JSON lines on stdout in the `benchflow-runner` format (`src/headless/parsers/runner.rs` lists every field). The scripts are embedded in the binary (`src/runners.rs`); a manifest writes one into the agent's home with `[[headless.files]] runner = "<file>"` and runs it with the framework's interpreter.

| Runner | Agent | Framework |
|---|---|---|
| `run_openhands_sdk.py` | `openhands-sdk` | openhands-sdk, openhands-tools |
| `run_nemo_agent.py` | `nemo-agent` | nvidia-nat |
| `run_strands.py` | `strands` | strands-agents, with the user's factory |
| `run_deerflow.py` | `deerflow` | deerflow-harness |
| `run_antigravity_sdk.py` | `antigravity-sdk` | google-antigravity |
| `run_langgraph.py` | `langgraph` | langgraph, with the user's graph |
| `run_eve.mjs` | `eve` | the user's Eve app |

The Python runners share `bf_events.py`: the event writer (it keeps stdout for events and sends everything else the framework prints to stderr), the error classification (`login`, `usage_limit`, `provider`, `model`, `setup`, `agent`), and exit codes (0 done, 1 failed, 2 the user must give the agent something).

The scripts were written for BenchFlow. Where an agent's setup follows another project's open-source adapter for the same agent (Apache-2.0), the script's docstring says which parts and where they come from.

## Trying a runner without a model key

`tests/mock_model.py` is a scripted OpenAI-compatible model: it calls a shell-like tool once, then answers, and returns OpenAI's 401 for the key `bad-key`. With the framework installed in a virtual environment:

```sh
python3 tests/mock_model.py 18765 &
mkdir -p /tmp/try/home/.benchflow/runners /tmp/try/work
cp bf_events.py run_strands.py /tmp/try/home/.benchflow/runners/
echo "Create bf_mock_out.txt containing benchflow." > /tmp/try/instruction.md
cd /tmp/try/work && HOME=/tmp/try/home BENCHFLOW_MODEL=openai/mock-model \
  OPENAI_API_KEY=good-key OPENAI_BASE_URL=http://127.0.0.1:18765/v1 \
  STRANDS_FACTORY=my_factory.py:create_agent STRANDS_FACTORY_PATH=/path/to/factory \
  <venv>/bin/python /tmp/try/home/.benchflow/runners/run_strands.py /tmp/try/instruction.md
```

The events on stdout should be a `start`, a `tool_call` and its `tool_result`, a `message`, `usage`, and `done`; with `OPENAI_API_KEY=bad-key`, an `error` of kind `login`.
