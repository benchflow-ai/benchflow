"""Tests for the tree data model of the tree-native Rollout."""

from benchflow.trajectories.tree import RolloutTree, Step, branch_points, trajectory


def test_step_carries_an_id_and_a_free_data_dict() -> None:
    """A Step is one edge: an id plus free-form content; data defaults empty."""
    bare = Step(id="s1")
    assert bare.id == "s1"
    assert bare.data == {}

    rich = Step(id="s2", data={"reasoning": "think", "action": {"tool": "ls"}})
    assert rich.data["action"]["tool"] == "ls"


def test_single_node_tree_has_empty_trajectory() -> None:
    """A fresh tree is one root node; the root-to-leaf path has no Steps."""
    tree = RolloutTree()
    assert trajectory(tree.root) == []


def test_advance_adds_a_child_wired_to_its_parent() -> None:
    """advance() grows the tree by one edge: a new child reached via a Step."""
    tree = RolloutTree()
    step = Step(id="s1")
    child = tree.advance(tree.root, step)

    assert child in tree.root.children
    assert child.parent is tree.root
    assert child.step_in is step


def test_trajectory_of_a_linear_rollout_is_the_whole_step_sequence_in_order() -> None:
    """In a degree-1 tree, trajectory() yields every Step root-to-leaf, in order."""
    tree = RolloutTree()
    s1, s2, s3 = Step(id="s1"), Step(id="s2"), Step(id="s3")
    n1 = tree.advance(tree.root, s1)
    n2 = tree.advance(n1, s2)
    leaf = tree.advance(n2, s3)

    assert trajectory(leaf) == [s1, s2, s3]


def test_a_linear_rollout_has_no_branch_points() -> None:
    """Every node in a degree-1 tree has at most one child — no branch points."""
    tree = RolloutTree()
    n1 = tree.advance(tree.root, Step(id="s1"))
    tree.advance(n1, Step(id="s2"))

    assert branch_points(tree) == []


def test_second_child_makes_the_parent_a_branch_point() -> None:
    """advance() on a node that already has a child turns it into a branch point."""
    tree = RolloutTree()
    n1 = tree.advance(tree.root, Step(id="s1"))
    tree.advance(n1, Step(id="s2a"))
    tree.advance(n1, Step(id="s2b"))

    assert branch_points(tree) == [n1]


def test_find_locates_a_node_by_id() -> None:
    """find() returns the node carrying the given id, anywhere in the tree."""
    tree = RolloutTree()
    n1 = tree.advance(tree.root, Step(id="s1"))
    n2 = tree.advance(n1, Step(id="s2"))

    assert tree.find(n2.id) is n2
    assert tree.find(tree.root.id) is tree.root


def test_find_returns_none_for_an_unknown_id() -> None:
    """find() returns None when no node carries the id."""
    tree = RolloutTree()
    assert tree.find("nonexistent") is None


def test_root_has_no_incoming_step() -> None:
    """The root is the start state; it has no parent and no incoming edge."""
    tree = RolloutTree()
    assert tree.root.parent is None
    assert tree.root.step_in is None


def test_node_carries_free_state_metadata() -> None:
    """Each node holds a state dict for environment/agent metadata."""
    tree = RolloutTree()
    child = tree.advance(tree.root, Step(id="s1"))
    child.state["cwd"] = "/repo"
    assert tree.find(child.id).state["cwd"] == "/repo"  # type: ignore[union-attr]


def test_branched_trajectories_share_a_prefix_and_diverge_after_the_branch() -> None:
    """Two sibling leaves yield trajectories equal up to the branch, then differ."""
    tree = RolloutTree()
    s1 = Step(id="s1")
    n1 = tree.advance(tree.root, s1)
    left = tree.advance(n1, Step(id="left"))
    right = tree.advance(n1, Step(id="right"))

    traj_left = trajectory(left)
    traj_right = trajectory(right)
    assert traj_left[0] is traj_right[0] is s1
    assert traj_left[-1].id == "left"
    assert traj_right[-1].id == "right"


def test_nodes_yields_every_node_including_root() -> None:
    """nodes() walks the whole tree — root plus every descendant."""
    tree = RolloutTree()
    n1 = tree.advance(tree.root, Step(id="s1"))
    left = tree.advance(n1, Step(id="left"))
    right = tree.advance(n1, Step(id="right"))

    assert set(tree.nodes()) == {tree.root, n1, left, right}
