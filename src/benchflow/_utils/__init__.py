"""benchflow._utils — small periphery I/O glue, private.

Holds small (<200 LOC) periphery modules that translate between external
artifacts (YAML files, git repos, scaffolded task dirs) and benchflow
shapes.

Members:
    yaml_loader      — YAML → RolloutConfig/EvaluationConfig
    benchmark_repos  — clone benchmark repos
    task_authoring   — init_task / check_task scaffolding
"""
