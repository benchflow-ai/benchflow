"""verl dataset class for the parquet ``bench train convert --format verl-sft`` writes.

Give it to verl's SFT trainer by path::

    data.custom_cls.path=<this file> data.custom_cls.name=BenchFlowSFTDataset

It imports only the standard library and verl, so verl loads it without benchflow installed.
Everything is verl's ``MultiTurnSFTDataset`` except two things it cannot do with agent rollouts:

- Parquet gives nested values one schema per column, so tool-call arguments and tool schemas of
  different shapes would gain each other's keys as nulls, which the chat template renders. The
  parquet therefore holds ``messages`` and ``tools`` as JSON strings, decoded here.
- verl trains every assistant message. ``train_message_indices`` lists the ones BenchFlow trains;
  the other assistant turns (a call cut at the output cap, one whose tool input was not a JSON
  object) keep loss mask 0 and stay as context.
"""

from __future__ import annotations

import json
from typing import Any

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

TRAIN_INDICES_KEY = "train_message_indices"


def _decoded(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class BenchFlowSFTDataset(MultiTurnSFTDataset):
    _trained: set[int] | None = None

    def _read_files_and_process(self) -> None:
        super()._read_files_and_process()
        if self.tools is not None:
            self.tools = [_decoded(tools) for tools in self.tools]

    def _build_messages(self, example: dict) -> list:
        example = dict(example)
        example[self.messages_key] = _decoded(example[self.messages_key])
        trained = example.get(TRAIN_INDICES_KEY)
        self._trained = None if trained is None else {int(index) for index in trained}
        return super()._build_messages(example)

    def _process_single_message(
        self, index: int, message: dict, *args: Any, **kwargs: Any
    ):
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index, message, *args, **kwargs
        )
        if self._trained is not None and index not in self._trained:
            loss_mask = loss_mask * 0
        return input_ids, loss_mask, attention_mask, inputs
