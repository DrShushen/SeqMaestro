from typing import Iterable

import pytest
from omegaconf import OmegaConf

try:
    import networkx as nx
except ImportError:
    nx = None

from igsr.tree_search.problem import Problem
from igsr.tree_search.session import SearchSession

# The search-tree bookkeeping graph is networkx-backed (optional; see the [benchmarks] extra).
# The core search-behaviour tests run on a base install; tests that inspect `session.graph`
# are skipped when networkx isn't available.
requires_networkx = pytest.mark.skipif(
    nx is None, reason="tree-graph inspection needs networkx (pip install -e .[benchmarks])"
)


def _mcts_extras(seed: int) -> dict:
    """Build the ``extras_dict`` that ``SearchSession.mcts`` needs for its token / wall-clock
    early-stop checks (all disabled here, so the search runs to its budget)."""
    return {
        "cfg": OmegaConf.create(
            {"experiment": {"break_on_tokens": False, "break_above_N_tokens": None, "break_on_wallclock": False}}
        ),
        "compute_profiler": None,
        "start_time": 0.0,
        "seed": seed,
        "logger": None,
    }


class LineWorld(Problem[int, float]):
    """A 1-D line world where the agent moves toward a target integer value.

    The optimal state is the *goal* value. The evaluation is the absolute
    distance to that goal (so lower is better). The reward used by MCTS is the
    negative evaluation as provided by the default ``Problem.reward`` helper.
    """

    def __init__(self, goal: int = 5):
        self.goal = goal

    # ------------------------------------------------------------------
    # Mandatory interface implementation
    # ------------------------------------------------------------------
    def successors(self, s: int) -> Iterable[int]:
        """Return the two neighbours along the line."""
        yield s + 1
        yield s - 1

    def evaluate(self, s: int) -> float:  # type: ignore[override]
        return abs(self.goal - s)

    def is_terminal(self, s: int) -> bool:  # type: ignore[override]
        return s == self.goal


# ======================================================================
# 1. Core search behaviour
# ======================================================================


def test_reaches_goal():
    """MCTS should eventually find the optimal state when the depth and budget
    are large enough.
    """
    problem = LineWorld(goal=7)
    session: SearchSession[int, float] = SearchSession(problem)

    best_val, best_state = session.mcts(
        start=0,
        depth_limit=20,
        total_budget=100,
        rollout_depth=10,
        seed=123,
        extras_dict=_mcts_extras(123),
    )

    assert best_state == 7  # the goal was found
    assert best_val == 0  # evaluation at the goal is zero


def test_depth_limit_blocks_goal():
    """If the depth-limit is lower than the distance to the goal, MCTS should
    not be able to reach it.
    """
    problem = LineWorld(goal=10)
    session: SearchSession[int, float] = SearchSession(problem)

    # depth_limit < distance(start, goal)
    best_val, best_state = session.mcts(
        start=0,
        depth_limit=5,
        total_budget=200,
        rollout_depth=10,
        seed=123,
        extras_dict=_mcts_extras(123),
    )

    # The goal is out of reach; the best state cannot be the goal and the
    # evaluation is strictly positive.
    assert best_state != 10
    assert best_val > 0


# ======================================================================
# 2. Non-hashable / complex state handling (exercises Problem.key)
# ======================================================================


class NonHashableListProblem(Problem[list[int], float]):
    """Problem whose states are *lists* (mutable & unhashable).

    The goal is to reach a single-element list whose item equals ``goal``.
    """

    def __init__(self, goal: int):
        self.goal = goal

    # Mandatory API -----------------------------------------------------
    def successors(self, s: list[int]) -> Iterable[list[int]]:  # type: ignore[override]
        x = s[0]
        return [[x + 1], [x - 1]]  # create *new* list objects

    def evaluate(self, s: list[int]) -> float:  # type: ignore[override]
        return abs(self.goal - s[0])

    def is_terminal(self, s: list[int]) -> bool:  # type: ignore[override]
        return s[0] == self.goal

    # Helpers -----------------------------------------------------------
    def key(self, s: list[int]):  # type: ignore[override]
        """Provide a hashable canonical representation."""
        return s[0]


def test_non_hashable_state_handling():
    """MCTS should cope with states that are not hashable by default (via ``key``)."""

    problem = NonHashableListProblem(goal=4)
    session: SearchSession[list[int], float] = SearchSession(problem)

    best_val, best_state = session.mcts(
        start=[0],
        depth_limit=10,
        total_budget=100,
        rollout_depth=10,
        seed=123,
        extras_dict=_mcts_extras(123),
    )

    assert best_state == [4]
    assert best_val == 0


# ======================================================================
# 3. Custom reward (overriding Problem.reward)
# ======================================================================


class BinaryRewardProblem(LineWorld):
    """Reward is 1 at the goal, 0 elsewhere (instead of −evaluate)."""

    def reward(self, s: int) -> float:  # type: ignore[override]
        return 1.0 if self.is_terminal(s) else 0.0


def test_mcts_custom_reward():
    """MCTS should work when ``reward`` is *not* −evaluate(s)."""

    problem = BinaryRewardProblem(goal=2)
    session: SearchSession[int, float] = SearchSession(problem)

    best_val, best_state = session.mcts(
        start=0,
        depth_limit=10,
        total_budget=200,
        rollout_depth=5,
        seed=123,
        extras_dict=_mcts_extras(123),
    )

    assert best_state == 2
    assert best_val == 0  # evaluation at the goal is still 0


# ======================================================================
# 4. Visualizer hook integration
# ======================================================================


class DummyVisualizer:
    """Collects calls for assertions without any graphical output."""

    def __init__(self):
        self.reset_called = 0
        self.new_nodes: list[int] = []

    def reset(self):
        self.reset_called += 1

    def on_new_node(self, node_id, parent_id, label_generator, state, best_state, best_val):
        self.new_nodes.append(node_id)


@requires_networkx
def test_visualizer_callbacks():
    problem = LineWorld(goal=1)
    viz = DummyVisualizer()
    session: SearchSession[int, float] = SearchSession(problem, visualiser=viz)

    best_val, best_state = session.mcts(
        start=0,
        depth_limit=3,
        total_budget=10,
        rollout_depth=2,
        seed=42,
        extras_dict=_mcts_extras(42),
    )

    # One reset at construction, one at the beginning of mcts().
    assert viz.reset_called == 2
    # Every node added to the graph should have triggered exactly one on_new_node callback.
    assert len(viz.new_nodes) == session.graph.number_of_nodes()
    # And the goal should have been found.
    assert best_state == 1


# ======================================================================
# 5. Session.reset() correctness
# ======================================================================


@requires_networkx
def test_session_reset_clears_state():
    problem = LineWorld(goal=3)
    session: SearchSession[int, float] = SearchSession(problem)

    session.mcts(
        start=0,
        depth_limit=5,
        total_budget=20,
        rollout_depth=2,
        seed=42,
        extras_dict=_mcts_extras(42),
    )

    # Sanity: something happened
    assert session.iteration > 0
    assert session.graph.number_of_nodes() > 0
    assert session.best_state is not None

    # Now reset and check everything is cleared
    session.reset()
    assert session.iteration == 0
    assert session.graph.number_of_nodes() == 0
    assert session.best_state is None
    assert session.best_val == float("inf")


# ======================================================================
# 6. Early-exit behaviour (stop_on_terminal)
# ======================================================================


@requires_networkx
def test_stop_on_terminal_true_exits_immediately():
    """If the start state is already terminal, MCTS should stop right away when
    ``stop_on_terminal=True`` (default): only the root node is visited and the
    iteration counter stays at 0.
    """

    problem = LineWorld(goal=0)  # start=0 is already the goal
    session: SearchSession[int, float] = SearchSession(problem)

    best_val, best_state = session.mcts(
        start=0,
        depth_limit=5,
        total_budget=50,
        rollout_depth=5,
        stop_on_terminal=True,
        seed=99,
        extras_dict=_mcts_extras(99),
    )

    assert best_state == 0 and best_val == 0
    assert session.graph.number_of_nodes() == 1  # only root
    assert session.iteration == 0  # exited before first increment


@requires_networkx
def test_mcts_continues_iterations_when_stop_on_terminal_false():
    """MCTS keeps iterating but does not expand additional nodes because the
    root is terminal. The iteration counter reaches ``total_budget-1``.
    """

    total_budget = 25
    problem = LineWorld(goal=0)
    session: SearchSession[int, float] = SearchSession(problem)

    best_val, best_state = session.mcts(
        start=0,
        depth_limit=5,
        total_budget=total_budget,
        rollout_depth=2,
        stop_on_terminal=False,
        seed=21,
        extras_dict=_mcts_extras(21),
    )

    assert best_state == 0 and best_val == 0
    # No expansion beyond root because it's terminal
    assert session.graph.number_of_nodes() == 1
    # Iteration counter ends at total_budget - 1 (loop index)
    assert session.iteration == total_budget - 1
