"""What a language model is told about the columns of the rows it writes code for.

Generated code takes one DataFrame, `X`, and indexes its columns by name: the sequence column and,
where the dataset declares them, the covariates. A model can use a covariate well only if it knows
what the covariate measures and what values it takes, so every generator that shows a model the
rows also tells it these facts. `describe_columns` gathers them, and `render_columns` puts them
into the words the proposer shows in its prompt. DEFT and IGSR are handed the same descriptions
with their rows (`seqmaestro.pipeline.vendored.handover`) and word them in their prompts, so every
generator is told what each covariate measures.

The descriptions come from the dataset's declaration (`Dataset.covariates`), and the lengths and
ranges from the rows the model is shown, so the code assumes nothing about a dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from seqmaestro.features.base import Inputs, as_inputs


def describe_columns(inputs, sequence_column: str,
                     descriptions: Mapping[str, str] | None) -> dict[str, Any]:
    """The columns of these rows as facts: the sequence column's length (one number when every row
    has the same length, and a range otherwise), and each covariate's description and range.

    `descriptions` maps each covariate of the rows to what it measures. Rows with a covariate that
    has no description are refused, because a model shown a bare name would guess what it means.
    """
    inputs: Inputs = as_inputs(inputs)
    described = dict(descriptions or {})
    missing = [name for name in inputs.covariate_columns if name not in described]
    if missing:
        raise ValueError(f"The rows carry covariate(s) {missing} with no description. Pass the "
                         f"dataset's (`Dataset.covariates`), so the model is told what each measures.")
    lengths = inputs.sequences.str.len()
    shortest, longest = int(lengths.min()), int(lengths.max())
    return {
        "sequence": {"name": sequence_column,
                     "length": shortest if shortest == longest else {"min": shortest, "max": longest}},
        "covariates": [
            {"name": name, "description": described[name],
             "min": float(inputs.covariates[name].min()), "max": float(inputs.covariates[name].max())}
            for name in inputs.covariate_columns
        ],
    }


def render_columns(columns: Mapping[str, Any]) -> str | None:
    """The facts of `describe_columns` in words, for a prompt, or None where they would add nothing
    to the task description: rows that all have one length and no covariates."""
    sequence, covariates = columns["sequence"], columns["covariates"]
    length = sequence["length"]
    if isinstance(length, int) and not covariates:
        return None
    extent = (f"{length:,} characters in every row" if isinstance(length, int)
              else f"{length['min']:,} to {length['max']:,} characters, varying by row")
    lines = ["Each row has these columns:",
             f"  - {sequence['name']!r} (text, {extent}): the sequence"]
    lines += [f"  - {covariate['name']!r} (a number, {covariate['min']:g} to {covariate['max']:g} "
              f"on the training rows): {covariate['description']}" for covariate in covariates]
    return "\n".join(lines)
