from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from deft.utils.feature import FeatureInfo

T = TypeVar("T")


class Selector(ABC):
    """Abstract base class for selection strategies."""

    @abstractmethod
    def select(self, population: list[T], k: int = 1) -> list[T]:
        """
        Select k items from the population.

        Args:
            population: List of items to select from
            k: Number of items to select

        Returns:
            List of selected items
        """


class TournamentSelector(Selector):
    """Tournament selection implementation.

    Draws from its own generator rather than the global `random` module, so that a run varies
    because its seed varied and for no other reason.
    """

    def __init__(self, tournament_size: int = 2, random_state: int | None = 42):
        """Initialize the instance."""
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(random_state)

    def select(self, population: list[FeatureInfo], k: int = 1) -> list[FeatureInfo]:
        """Select values."""
        selected = []
        for _ in range(k):
            # Select random candidates for tournament
            indices = self.rng.choice(
                len(population), size=min(self.tournament_size, len(population)), replace=False
            )
            candidates = [population[i] for i in indices]
            # Select the candidate with highest score
            winner = max(candidates, key=lambda x: x.score)
            selected.append(winner)
        return selected


class RankSelector(Selector):
    """Rank-based selection implementation.

    Draws from its own generator rather than the global numpy RNG, for the same reason as
    `TournamentSelector`.
    """

    def __init__(self, random_state: int | None = 42):
        """Initialize the instance."""
        self.rng = np.random.default_rng(random_state)

    def select(self, population: list[FeatureInfo], k: int = 1) -> list[FeatureInfo]:
        """Select values."""
        N = len(population)
        # Sort population by score in descending order
        sorted_pop = sorted(population, key=lambda x: x.score, reverse=True)

        # Calculate selection probabilities
        ranks = np.arange(1, N + 1)
        probabilities = 1 / (ranks + N)
        probabilities = probabilities / probabilities.sum()  # Normalize

        # Select k items based on rank probabilities
        selected_indices = self.rng.choice(N, size=k, p=probabilities, replace=True)

        return [sorted_pop[i] for i in selected_indices]


class TopKSelector(Selector):
    """Select the top k items based on score."""

    def select(self, population: list[FeatureInfo], k: int = 1) -> list[FeatureInfo]:
        """Select values."""
        return sorted(population, key=lambda x: x.score, reverse=True)[:k]
