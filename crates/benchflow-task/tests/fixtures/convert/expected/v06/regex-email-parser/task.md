Write a Python function `parse_email_addresses(text: str) -> list[dict]` that extracts all email addresses from a block of text and returns structured data. For each email, return: `{'local': str, 'domain': str, 'full': str}`. Handle edge cases: emails in angle brackets `<user@domain.com>`, emails with dots and hyphens in local part, emails with subdomains.

Do NOT match: strings without @ signs, emails with spaces, or emails ending with dots.

Example:
```python
text = 'Contact us at support@example.com or <sales@sub.domain.co.uk>'
parse_email_addresses(text)
# [{'local': 'support', 'domain': 'example.com', 'full': 'support@example.com'},
#  {'local': 'sales', 'domain': 'sub.domain.co.uk', 'full': 'sales@sub.domain.co.uk'}]
```

```toml task
[about]
author_name = "benchflow-skill-eval"
difficulty = "medium"
category = "skill-eval"
tags = ["skill-eval", "code-specialist"]

[agent]
timeout = "10m"

[sandbox]
cpus = 1
memory = "2 GB"
network = "open"
skills = "sandbox/skills"

[verifier]
timeout = "2m"

[import.v06]
schema_version = "1.3"
sandbox = { skills_dir = "/skills" }

[import.v06.verifier_md]
path = "verifier/verifier.md"
default_strategy = "judge"
strategies = { judge = "script" }
roles = ["reviewer"]
```
