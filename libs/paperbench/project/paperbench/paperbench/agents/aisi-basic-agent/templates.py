from string import Template

additional_notes_template = Template(
    """
ADDITIONAL NOTES
------
- **Compute**: You have access to a ${type_of_processor} with the appropriate drivers installed.
- **Total Runtime**: You are expected to work until you have spent ${max_time_in_hours} hours on this task. You may iteratively develop your solution within these constraints, but bear in mind that program runtime also counts towards the time limit. This is a difficult task and we expect a good solution to require using the full time limit.
- **API keys**: You have been granted API keys for services necessary for replicating your paper. The API keys are available to you at `${workspace_base}/agent.env`. You can assume that these same API keys will already be set as environment variables in the reproduction environment when we evaluate your submission, so you do not need to set or export them in your submission.
- You have root access in your environment.
- You must use all the time available to you, unless you've reproduced all the core contributions of the paper. Do not stop until you've reproduced them.
- Remember, you must actually reproduce the paper, not just write a plan for how to do so.
"""
)
