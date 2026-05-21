"""Tests for node-scoped scoring — the Reward plane's score(node) path.

The architecture's Reward contract is ``score(node) -> VerifyResult``. A
scorer examines a RolloutNode — the leaf for outcome reward, its root-to-leaf
path for process reward — and emits a RewardEvent tagged (space, granularity).
"""

from benchflow.rewards.events import RewardEvent
from benchflow.rewards.node import score_node
from benchflow.rewards.protocol import VerifyResult
from benchflow.trajectories.tree import RolloutNode, RolloutTree, Step, trajectory


class OutcomeScorer:
    """Reads the leaf node's state for a terminal output-space reward."""

    def __init__(self, source: str = "outcome") -> None:
        self.source = source

    async def score(self, node: RolloutNode) -> RewardEvent:
        return RewardEvent(
            type="terminal",
            reward=float(node.state.get("reward", 0.0)),
            source=self.source,
            space="output",
            granularity="terminal",
        )


class PathLengthScorer:
    """A process-space scorer — rewards based on the node's trajectory length."""

    source = "path_length"

    async def score(self, node: RolloutNode) -> RewardEvent:
        steps = trajectory(node)
        return RewardEvent(
            type="process",
            reward=float(len(steps)),
            source=self.source,
            space="action",
            granularity="terminal",
        )


async def test_score_node_runs_a_scorer_and_returns_verify_result():
    node = RolloutNode(id="leaf", state={"reward": 1.0})
    result = await score_node(node, [OutcomeScorer()])
    assert isinstance(result, VerifyResult)
    assert result.reward == 1.0
    assert len(result.events) == 1


async def test_score_node_records_each_scorer_in_items():
    node = RolloutNode(id="leaf", state={"reward": 0.8})
    result = await score_node(node, [OutcomeScorer("a"), OutcomeScorer("b")])
    assert set(result.items) == {"a", "b"}
    assert result.items["a"] == 0.8


async def test_score_node_reward_is_the_output_space_signal():
    """The headline reward is the Output space — not a process-space event."""
    node = RolloutNode(id="leaf", state={"reward": 1.0})
    result = await score_node(node, [PathLengthScorer(), OutcomeScorer()])
    assert result.reward == 1.0  # the output-space event, not the path length
    assert any(e.space == "action" for e in result.events)  # process rides along


async def test_process_scorer_reads_the_node_trajectory():
    tree = RolloutTree()
    n1 = tree.advance(tree.root, Step(id="s1"))
    n2 = tree.advance(n1, Step(id="s2"))
    result = await score_node(n2, [PathLengthScorer()])
    assert result.items["path_length"] == 2.0  # two Steps on the root→n2 path


async def test_score_node_with_no_scorers_is_zero():
    result = await score_node(RolloutNode(id="leaf"), [])
    assert result.reward == 0.0
    assert result.events == []


async def test_score_node_tags_result_with_space_and_granularity():
    """The VerifyResult carries the (space, granularity) tag the spec mandates."""
    node = RolloutNode(id="leaf", state={"reward": 1.0})
    result = await score_node(node, [OutcomeScorer()])
    assert result.space == "output"
    assert result.granularity == "terminal"


async def test_score_node_signals_when_no_outcome_scorer_ran():
    """reward=0.0 with NO output-space scorer is 'nobody scored', not 'scored 0'.

    SHOULD-FIX 7: a node scored only by a process-space scorer has no outcome
    signal — score_node must distinguish that from an honest zero by
    populating VerifyResult.error.
    """
    node = RolloutNode(id="leaf")
    result = await score_node(node, [PathLengthScorer()])  # process-space only
    assert result.reward == 0.0
    assert result.error is not None  # nobody scored the outcome
    assert "nobody scored" in result.error


async def test_score_node_honest_zero_has_no_error():
    """An output-space scorer reporting 0.0 is an honest zero — error stays None."""
    node = RolloutNode(id="leaf", state={"reward": 0.0})
    result = await score_node(node, [OutcomeScorer()])
    assert result.reward == 0.0
    assert result.error is None  # a real scorer ran and said 0.0


async def test_score_node_no_scorers_at_all_signals_nobody_scored():
    """Empty scorer list: no outcome event either, so error is populated."""
    result = await score_node(RolloutNode(id="leaf"), [])
    assert result.error is not None


async def test_score_node_same_source_items_collision_last_wins():
    """SHOULD-FIX 9: two scorers sharing a source name — last wins in items.

    items is keyed by source, so a name collision means the last scorer's
    reward lands in items[source]. Both events are still kept in events, so
    no signal is lost — this test pins that semantics.
    """
    node = RolloutNode(id="leaf", state={"reward": 0.3})

    class FixedScorer:
        def __init__(self, source: str, reward: float) -> None:
            self.source = source
            self._reward = reward

        async def score(self, n: RolloutNode) -> RewardEvent:
            return RewardEvent(
                type="terminal",
                reward=self._reward,
                source=self.source,
                space="output",
            )

    result = await score_node(
        node, [FixedScorer("dup", 0.2), FixedScorer("dup", 0.9)]
    )
    # items collapses to one entry — the last scorer wins
    assert result.items == {"dup": 0.9}
    # both events survive — no signal lost
    assert [e.reward for e in result.events] == [0.2, 0.9]
