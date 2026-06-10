---
version: '1.0'
metadata:
  author_name: benchflow-skill-eval
  difficulty: medium
  category: skill-eval
  tags:
  - skill-eval
  - code-specialist
agent:
  timeout_sec: 600
verifier:
  timeout_sec: 120
environment:
  cpus: 1
  memory_mb: 2048
  allow_internet: true
  skills_dir: /skills
---

## prompt

Implement a function `topo_sort(graph: dict[str, list[str]]) -> list[str]` that performs topological sorting on a directed graph. The graph is represented as an adjacency list (node → list of nodes it points to). If the graph contains a cycle, raise a ValueError with the message 'Graph contains a cycle'. The function should be deterministic — for nodes with equal ordering, sort alphabetically.

Example:
```python
graph = {'a': ['b', 'c'], 'b': ['d'], 'c': ['d'], 'd': []}
topo_sort(graph)  # ['a', 'b', 'c', 'd'] or ['a', 'c', 'b', 'd']

cyclic = {'a': ['b'], 'b': ['c'], 'c': ['a']}
topo_sort(cyclic)  # raises ValueError('Graph contains a cycle')
```

Write tests to verify correctness.
