# Skill Eval Rubric

Skill: `code-specialist`

Case: `regex-email-parser`

The verifier scores the agent from 0.0 to 1.0 using the rubric in `case.json` and the observed trajectory under `/logs/agent`.

Expected behavior:

- The agent produced a working parse_email_addresses function
- The regex handles standard emails, angle brackets, and subdomains
- Invalid emails are correctly rejected
- The agent wrote tests covering edge cases
- The function returns the structured dict format
