"""benchflow._utils — small periphery I/O glue, private.

Holds small (<200 LOC) periphery modules that translate between external
artifacts (YAML files, git repos, scaffolded task dirs) and benchflow
shapes. NOT imported by ``benchflow.contracts``, ``benchflow.trial``, or
``benchflow.job`` core paths — only by orchestrator setup, CLI, and
public ``__init__.py`` re-exports.

Members today:
    yaml_loader      — YAML → TrialConfig/JobConfig (was trial_yaml.py)
    benchmark_repos  — clone benchmark repos (was task_download.py)
    task_authoring   — init_task / check_task scaffolding (was tasks.py)

Promotion criteria — a `_utils/` member earns its own top-level module or
subpackage when it (a) crosses 200 LOC, (b) gains a programmatic public
consumer, or (c) acquires a second sibling that shares its specific axis.

This rule is what keeps `_utils/` from becoming the §8.4 junk drawer.
Without enforcement in PR review, it will drift; with enforcement, it
captures real periphery work without inventing false domains.
"""
