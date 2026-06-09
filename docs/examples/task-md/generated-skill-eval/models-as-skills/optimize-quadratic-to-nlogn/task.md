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

The following function finds all pairs in a list that sum to a target value. It works but is O(n²). Optimize it to O(n) or O(n log n) while maintaining the same output format (sorted list of pairs, each pair sorted).

```python
def find_pairs(nums: list[int], target: int) -> list[tuple[int, int]]:
    result = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                pair = tuple(sorted([nums[i], nums[j]]))
                if pair not in result:
                    result.append(pair)
    return sorted(result)
```

Your optimized version must produce identical output for all inputs. Include a brief explanation of the complexity improvement.
