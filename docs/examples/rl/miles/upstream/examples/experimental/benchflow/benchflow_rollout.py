"""Rollout function for the BenchFlow example.

``RolloutFn`` (``--rollout-function-path benchflow_rollout.RolloutFn``) is the
default rollout function plus per-step metrics about the episodes that entered
training (``benchflow_agent_function.benchflow_metrics``): the share of each
``exit_status``, the share of replies cut at ``max_tokens``, episodes that ran
out of context, integrity flags, and mean turns, tool calls, tokens and times.
"""

import logging

from benchflow_agent_function import benchflow_metrics

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn

logger = logging.getLogger(__name__)


class RolloutFn(InferenceRolloutFn):
    """The default rollout function, plus BenchFlow's per-step metrics."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)
        samples = [s for group in output.samples for s in (group if isinstance(group, list) else [group])]
        metrics = benchflow_metrics(samples)
        if metrics:
            output.metrics = {**(output.metrics or {}), **metrics}
            logger.info(f"BenchFlow metrics for rollout {input.rollout_id}: {metrics}")
        return output
