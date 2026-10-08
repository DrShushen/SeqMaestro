"""Group the reduced models' features into concepts, across arms.

Different arms name the same idea differently. One writes a fraction of a base class over a stretch
of the window, another computes the same quantity under a different name over a slightly different
range, and a third splits on a single base inside that stretch. Counting names would report these
as three separate findings. Several arms are asked the same question so that agreement like this
counts as one finding that three arms reached.

Features are therefore grouped **by what they compute**, not by what they are called. Every feature
is evaluated on one common set of rows, and two features join the same concept when their values
agree. The matcher, its correlation, its threshold and the way matched pairs are chained into groups
all come from the harness (`seqmaestro.harness.stability`), so a concept here means the same as a
concept elsewhere in the project.

This module adds the join. Each concept carries the arms that reached it, the feature sources among
them, and the reliability evidence attached to its members, so a reader can rank ideas by weight of
evidence instead of ranking columns by importance.

The result has two limits, and any use of it should state them. Grouping is transitive, so a chain
of borderline matches can merge two distinct ideas. And the grouping is computed on the rows the
models were fitted and tuned on (D, train + validation). Those rows can show whether two columns
compute the same thing, but not whether either one holds. Nothing here reads the test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness.stability import FeatureRun, Matcher, consensus_labels
from seqmaestro.report import naming

#: A member's sign counts towards its concept's direction only when its marginal effect (a
#: rank-biserial correlation, or Spearman's rho for a continuous target) reaches this size, in
#: either direction. Below it, the sign is noise.
DIRECTION_FLOOR = 0.05


@dataclass
class Member:
    """One feature of one arm's reduced model, inside a concept."""

    arm: str
    feature: str
    kind: str
    description: str
    rank: int
    marginal_effect: float | None
    stability: float | None = None
    cluster_stability: float | None = None
    #: Tier 2's association on the held-out rows: an AUROC for a binary target, Spearman's rho for
    #: a continuous one (`facts["metrics"]["association"]`).
    association: float | None = None
    association_q: float | None = None
    necessity_p: float | None = None


@dataclass
class Concept:
    """A property one arm or several arrived at, whatever each called it. `association_centre` is
    the value of the association that means none: 0.5 for an AUROC, 0 for a rank correlation."""

    association_centre: float
    members: list[Member] = field(default_factory=list)

    @property
    def arms(self) -> list[str]:
        return sorted({member.arm for member in self.members})

    @property
    def sources(self) -> list[str]:
        return sorted({naming.parse(member.arm).source_key for member in self.members})

    @property
    def model_families(self) -> list[str]:
        return sorted({naming.parse(member.arm).model_key for member in self.members})

    @property
    def written_and_enumerated(self) -> bool:
        """Whether both a written feature and an enumerated one reached this concept.

        Agreement between two arms drawing on one pool of columns says less than agreement between
        two that had no vocabulary in common. The test takes each feature's kind and not its arm's
        pool, because DEFT on `llm_generated_joint` writes features at each split of its tree and
        also draws enumerated candidates, so its features can be of either kind.
        """
        return {member.kind == "llm_generated" for member in self.members} == {True, False}

    @property
    def representative(self) -> Member:
        """The member the concept is named and described by: whichever ranked highest.

        Name and description both come from this member. Taking the name from one member and the
        description from another produces rows where the two do not match, which looks like an error
        even when both are correct.
        """
        return min(self.members, key=lambda member: (member.rank, member.feature, member.arm))

    @property
    def label(self) -> str:
        return self.representative.feature

    @property
    def description(self) -> str:
        return self.representative.description

    @property
    def member_names(self) -> list[str]:
        return sorted({member.feature for member in self.members})

    @property
    def best_stability(self) -> float | None:
        found = [m.cluster_stability for m in self.members if m.cluster_stability is not None]
        return max(found) if found else None

    @property
    def best_association_q(self) -> float | None:
        found = [m.association_q for m in self.members if m.association_q is not None]
        return min(found) if found else None

    @property
    def strongest_association(self) -> float | None:
        """The member's association furthest from the value that means none, in either direction."""
        found = [m.association for m in self.members if m.association is not None]
        return max(found, key=lambda value: abs(value - self.association_centre)) if found else None

    @property
    def direction(self) -> str | None:
        """Which way the concept points, where its members agree on a sign: `positive` where higher
        values go with the positive class or a higher target, `negative` where they go with the
        other class or a lower target.

        Members are matched on absolute correlation, so a feature and its negation can share a
        concept. The sign is taken from each member's marginal effect, and a concept whose members
        disagree carries no direction rather than the majority's.
        """
        signs = {np.sign(member.marginal_effect) for member in self.members
                 if member.marginal_effect is not None
                 and abs(member.marginal_effect) >= DIRECTION_FLOOR}
        if len(signs) != 1:
            return None
        return "positive" if signs.pop() > 0 else "negative"

    def to_dict(self) -> dict:
        return {
            "label": self.label, "description": self.description, "arms": self.arms,
            "sources": self.sources, "model_families": self.model_families,
            "member_names": self.member_names, "direction": self.direction,
            "best_stability": self.best_stability, "best_association_q": self.best_association_q,
            "strongest_association": self.strongest_association,
            "written_and_enumerated": self.written_and_enumerated,
            "members": [vars(member) for member in self.members],
        }


@dataclass
class ArmColumns:
    """The columns the reduction kept for one arm: their records, their values on D, and what the
    report knows of each: its rank and marginal effect from Stage 3A, and its evidence from the
    panel, which holds only the columns the reduced model uses."""

    arm: str
    records: list[FeatureRecord]
    values: np.ndarray
    features: list[dict]
    evidence: dict[str, dict]


def build(arms: list[ArmColumns], matcher: Matcher, association: tuple[str, float]) -> list[Concept]:
    """Group the columns of `arms` into concepts, each carrying its members' evidence.
    `association` names Tier 2's association in the evidence and the value of it that means none
    (`seqmaestro.reliability.tier2.ASSOCIATION`)."""
    measure, centre = association
    runs = [FeatureRun(label=entry.arm, records=list(entry.records), values=entry.values)
            for entry in arms]
    labels = consensus_labels(runs, matcher)

    concepts: dict[int, Concept] = {}
    for entry, run, run_labels in zip(arms, runs, labels, strict=True):
        about = {feature["name"]: feature for feature in entry.features}
        for name, identifier in zip(run.names, run_labels, strict=True):
            feature, evidence = about[name], entry.evidence.get(name, {})
            concepts.setdefault(identifier, Concept(association_centre=centre)).members.append(Member(
                arm=entry.arm, feature=name, kind=feature["kind"],
                description=feature["description"], rank=feature["rank"],
                marginal_effect=feature["marginal_effect"],
                stability=evidence.get("pi_hat"), cluster_stability=evidence.get("cluster_pi_hat"),
                association=evidence.get(measure), association_q=evidence.get("q_bh"),
                necessity_p=evidence.get("p_holm")))
    return sorted(concepts.values(), key=evidence_order)


def evidence_order(concept: Concept) -> tuple:
    """The order concepts are presented in: by reach, then by strength of evidence.

    Reach (the number of arms) comes first, because properties that several independent arms
    arrived at are what this pipeline is built to surface. Among concepts of equal reach, the
    smaller best q-value on held-out rows comes first, then the higher best selection frequency,
    then the label.
    """
    return (-len(concept.arms),
            concept.best_association_q if concept.best_association_q is not None else 1.0,
            -(concept.best_stability or 0.0),
            concept.label)
