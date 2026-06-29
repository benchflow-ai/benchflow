---
schema_version: "1.3"
task:
  name: benchflow/multi-agent-adapter
  description: Schema-only fixture for uniform multi-agent workflow adapters and LiteLLM trajectory tracing
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [multi-agent, adapters, litellm, trajectory, task-md]
metadata:
  category: schema
  note: This fixture demonstrates the proposed benchflow.multi_agent authoring surface. It is not a runnable task package.
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 180
environment:
  cpus: 2
  memory_mb: 4096
agents:
  roles:
    orchestrator:
      agent: python-workflow
      model: litellm/gpt-5.5
      capabilities: [workflow-launch]
    planner:
      agent: external
      model: litellm/gpt-5.5
      capabilities: [planning]
    implementer:
      agent: external
      model: litellm/gpt-5.5
      capabilities: [code-edit, tool-use]
    reviewer:
      agent: external
      model: litellm/gpt-5.5
      capabilities: [review, critique]
scenes:
  - name: external-workflow
    turns:
      - role: orchestrator
benchflow:
  multi_agent:
    adapter: langgraph
    mode: external-workflow
    entrypoint: workflows.design_review:run
    workflow_root: workflow/
    trace:
      llm_proxy: litellm
      capture_raw_llm: required
      capture_framework_events: best-effort
      relationship_graph: required
      redact_messages: false
      emit:
        - trajectory/llm_raw.jsonl
        - trajectory/multiagent_events.jsonl
        - trajectory/agent_graph.json
        - trajectory/index.json
        - trajectory/redaction_report.json
    agents:
      mapping:
        planner:
          role: planner
          framework_node: plan_node
        implementer:
          role: implementer
          framework_node: implement_node
        reviewer:
          role: reviewer
          framework_node: review_node
    relationships:
      allowed:
        - delegates
        - reviews
        - handoff
        - parallel_child
        - fan_in
    adapters:
      langgraph:
        graph_object: design_review_graph
        node_role_map:
          plan_node: planner
          implement_node: implementer
          review_node: reviewer
      autogen:
        team_object: team
        participant_role_map:
          primary: implementer
          critic: reviewer
      crewai:
        crew_object: crew
        task_role_map:
          plan: planner
          implement: implementer
          review: reviewer
      generic-openai-compatible:
        command: python -m workflow.run
        attribution:
          fallback: litellm_virtual_key
---
# Multi-agent adapter fixture

This schema-only task demonstrates the proposed authoring surface for hosting an
external multi-agent workflow inside BenchFlow.

The orchestrator role launches the external workflow. The framework-native
agents are mapped back to BenchFlow roles so result viewers can show the
relationship between planner, implementer, and reviewer trajectories.

Success criteria for a runnable version of this task would be:

1. run the declared external workflow through the selected adapter;
2. route all model calls through the BenchFlow LiteLLM proxy;
3. emit raw LiteLLM call records to `trajectory/llm_raw.jsonl`;
4. emit normalized relationship-aware events to
   `trajectory/multiagent_events.jsonl`;
5. emit an agent/workflow DAG to `trajectory/agent_graph.json`;
6. score the final `/app` state with the normal verifier contract.
