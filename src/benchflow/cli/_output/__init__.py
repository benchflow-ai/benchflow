"""What ``bench`` prints at the end of a run.

:mod:`run_summary` builds the default final screen (one row per trial, one
footer, the results folder) and the ``--json`` document from an
``EvaluationResult``; the long-form footer (``cli._shared._report_eval_result``)
is what ``--verbose`` adds after it.
"""
