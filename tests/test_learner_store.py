"""Tests for the LearnerStore — the persistent, versioned store behind
continual-learning Jobs (capability 5).

A continual-learning Job runs in ``sequential-shared`` mode: Rollouts run in
order over one persistent learner store (memory + skills). Each rollout stamps
a generation; a regression in the learning curve can revert a generation. The
store is the one snapshot layer that deliberately does NOT roll back with a
Branch.
"""

import pytest

from benchflow.learner_store import Generation, LearnerState, LearnerStore

# --- LearnerState: the (memory + skills) payload ---


def test_learner_state_starts_empty():
    state = LearnerState()
    assert state.memory == {}
    assert state.skills == {}


def test_learner_state_copy_is_independent():
    state = LearnerState(memory={"a": 1}, skills={"s": "body"})
    clone = state.copy()
    clone.memory["a"] = 999
    clone.skills["s"] = "mutated"
    # The original is untouched — copy is deep.
    assert state.memory == {"a": 1}
    assert state.skills == {"s": "body"}


# --- the empty store ---


def test_store_starts_at_generation_zero():
    store = LearnerStore()
    assert store.generation == 0
    assert store.current().memory == {}
    assert store.current().skills == {}


def test_store_current_is_a_copy_not_the_live_state():
    """Reading current() must not hand out a mutable handle to internal state."""
    store = LearnerStore()
    snapshot = store.current()
    snapshot.memory["leak"] = True
    assert store.current().memory == {}


# --- commit: stamp a generation ---


def test_commit_advances_the_generation_counter():
    store = LearnerStore()
    state = store.current()
    state.memory["lesson"] = "always read the task.toml"
    gen = store.commit(state)
    assert gen == 1
    assert store.generation == 1
    assert store.current().memory == {"lesson": "always read the task.toml"}


def test_commit_returns_a_generation_record():
    store = LearnerStore()
    state = store.current()
    state.skills["greet"] = "say hi"
    gen = store.commit(state, metric=0.5)
    record = store.history[gen]
    assert isinstance(record, Generation)
    assert record.number == 1
    assert record.metric == 0.5
    assert record.state.skills == {"greet": "say hi"}


def test_commits_accumulate_across_generations():
    store = LearnerStore()
    s1 = store.current()
    s1.memory["g1"] = True
    store.commit(s1, metric=0.3)

    s2 = store.current()
    s2.memory["g2"] = True
    store.commit(s2, metric=0.6)

    assert store.generation == 2
    # Generation 2 carries forward generation 1's memory.
    assert store.current().memory == {"g1": True, "g2": True}


def test_commit_snapshots_state_defensively():
    """A commit must capture state by value — later mutation of the passed
    object must not leak into the committed generation."""
    store = LearnerStore()
    state = store.current()
    state.memory["x"] = 1
    store.commit(state)
    state.memory["x"] = 999  # mutate the caller's copy after commit
    assert store.current().memory == {"x": 1}


# --- revert: roll back a generation on regression ---


def test_revert_restores_a_prior_generation():
    store = LearnerStore()
    s1 = store.current()
    s1.memory["good"] = "keep me"
    store.commit(s1, metric=0.8)

    s2 = store.current()
    s2.memory["bad"] = "regression"
    store.commit(s2, metric=0.2)

    # Generation 2 regressed — revert to generation 1.
    store._revert(1)
    assert store.generation == 1
    assert store.current().memory == {"good": "keep me"}


def test_revert_to_generation_zero_clears_the_store():
    store = LearnerStore()
    s1 = store.current()
    s1.skills["s"] = "body"
    store.commit(s1)
    store._revert(0)
    assert store.generation == 0
    assert store.current().skills == {}


def test_revert_drops_later_history():
    store = LearnerStore()
    store.commit(store.current(), metric=0.1)
    store.commit(store.current(), metric=0.2)
    store.commit(store.current(), metric=0.3)
    store._revert(1)
    # Generations 2 and 3 are gone from history.
    assert set(store.history) == {0, 1}


def test_generation_numbers_are_monotonic_after_revert():
    """A reverted generation number is never reused — the next commit stamps
    a fresh number so each generation is a durable per-rollout stamp."""
    store = LearnerStore()
    store.commit(store.current(), metric=0.1)  # gen 1
    store.commit(store.current(), metric=0.2)  # gen 2
    store.commit(store.current(), metric=0.3)  # gen 3
    store._revert(1)
    # The next commit does NOT re-use 2 — it stamps 4.
    gen = store.commit(store.current(), metric=0.5)
    assert gen == 4
    assert store.generation == 4
    assert set(store.history) == {0, 1, 4}


def test_commit_or_revert_keeps_monotonic_numbering():
    """commit_or_revert that rejects a regression still advances the number
    space for the next genuine commit."""
    store = LearnerStore()
    store.commit(store.current(), metric=0.8)  # gen 1
    # A regression is rejected — no generation stamped.
    assert store.commit_or_revert(store.current(), metric=0.3) is False
    # The next improvement stamps gen 2 (the rejected one consumed nothing).
    assert store.commit_or_revert(store.current(), metric=0.9) is True
    assert store.generation == 2


def test_revert_to_unknown_generation_raises():
    store = LearnerStore()
    store.commit(store.current())
    with pytest.raises(ValueError, match="generation"):
        store._revert(5)


def test_revert_forward_raises():
    """Revert only goes back — reverting to a future generation is a bug."""
    store = LearnerStore()
    store.commit(store.current())
    with pytest.raises(ValueError, match="generation"):
        store._revert(2)


# --- the learning curve / regression helper ---


def test_learning_curve_is_metric_per_generation_in_order():
    store = LearnerStore()
    store.commit(store.current(), metric=0.2)
    store.commit(store.current(), metric=0.5)
    store.commit(store.current(), metric=0.4)
    assert store.learning_curve() == [0.2, 0.5, 0.4]


def test_regressed_detects_a_drop_against_the_best_so_far():
    store = LearnerStore()
    store.commit(store.current(), metric=0.5)
    store.commit(store.current(), metric=0.7)
    # 0.6 < best-so-far 0.7 => regression.
    assert store._regressed(0.6) is True


def test_regressed_false_when_metric_improves():
    store = LearnerStore()
    store.commit(store.current(), metric=0.5)
    assert store._regressed(0.9) is False


def test_regressed_false_on_an_empty_store():
    """With no prior generation there is nothing to regress against."""
    store = LearnerStore()
    assert store._regressed(0.0) is False


def test_regressed_respects_a_tolerance():
    """A tiny dip within tolerance is noise, not a regression."""
    store = LearnerStore()
    store.commit(store.current(), metric=0.80)
    assert store._regressed(0.79, tolerance=0.05) is False
    assert store._regressed(0.70, tolerance=0.05) is True


# --- commit_or_revert: the continual-learning step ---


def test_commit_or_revert_keeps_an_improvement():
    store = LearnerStore()
    store.commit(store.current(), metric=0.4)
    state = store.current()
    state.memory["new"] = "lesson"
    kept = store.commit_or_revert(state, metric=0.7)
    assert kept is True
    assert store.generation == 2
    assert store.current().memory == {"new": "lesson"}


def test_commit_or_revert_rejects_a_regression():
    store = LearnerStore()
    base = store.current()
    base.memory["keep"] = "this"
    store.commit(base, metric=0.7)

    regressed = store.current()
    regressed.memory["bad"] = "drop this"
    kept = store.commit_or_revert(regressed, metric=0.3)

    assert kept is False
    # The store stayed at generation 1 with the good state.
    assert store.generation == 1
    assert store.current().memory == {"keep": "this"}


def test_commit_or_revert_on_empty_store_always_commits():
    store = LearnerStore()
    kept = store.commit_or_revert(store.current(), metric=0.0)
    assert kept is True
    assert store.generation == 1
