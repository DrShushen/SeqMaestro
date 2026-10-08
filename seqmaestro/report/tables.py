"""Every table a report may include, rendered from `facts.json` in a reader's names.

A report source includes a table with `{{table:NAME}}` and says what it means, and never retypes a
number. This module decides which tables exist, and assembling a report lists any table the source
left out.

Table names are fixed, except the per-arm ones: `features_<arm>` for every reduced arm whose
reduction kept a column and `evidence_<arm>` for every profiled one, where `<arm>` is
`model__pool`. The drawn symbolic model's equation is a table file too (`equation`), built from its
exported internals.
"""

from __future__ import annotations

import re
from typing import Any

from seqmaestro.report import facts as facts_module
from seqmaestro.report import markdown as md
from seqmaestro.report import naming
from seqmaestro.report.concepts import Concept
from seqmaestro.report.internals import NO_FEATURES
from seqmaestro.target import BINARY, CONTINUOUS

#: A feature kept by at least this many reduced arms, by name, is listed as shared.
SHARED_AT_LEAST = 3

#: What a table calls the columns whose meaning differs between the kinds of target, and how it
#: prints them. A feature's association with the target on its own (its marginal effect, and Tier 2
#: on the held-out rows) is a separation of the classes for a binary target and a rank correlation
#: for a continuous one. The cost of removing a feature (Tier 3) is the tested change in log-loss
#: for a binary target, and for a continuous one the fall in R², which is the tested change in
#: squared error divided by the held-out target's variance: the same test on an axis that does not
#: depend on the target's units.
WORDS = {
    BINARY: {"direction": "argues for, before reduction", "concept_direction": "higher values argue for",
             "alone": "separation on its own", "held_out": "separation, held out",
             "held_out_format": md.score, "cost": ("delta_log_loss", "cost of removing it")},
    CONTINUOUS: {"direction": "higher values, before reduction", "concept_direction": "higher values",
                 "alone": "rank correlation on its own", "held_out": "rank correlation, held out",
                 "held_out_format": md.signed, "cost": ("delta_r2", "cost of removing it, in R²")},
}


def headline_order(target_kind: str) -> tuple[str, ...]:
    """The order the headline set is presented in: interpretable model classes first, the best
    enumerated-feature arm next, the reference model last."""
    return (*(f"best_{family}" for family in facts_module.HEADLINE_FAMILIES[target_kind]),
            "best_enumerated", "black_box_reference")


def direction_word(direction: str | None, directions: dict[str, Any]) -> str:
    """A recorded direction as a report prints it (`facts["directions"]`)."""
    return directions["printed"].get(direction, md.MISSING)


def _model_source(arm: str) -> list[str]:
    configuration = naming.parse(arm)
    return [configuration.model_short, configuration.source_short]


def build(facts: dict[str, Any], concepts: list[Concept]) -> dict[str, str]:
    """Every table, by name."""
    kind, directions = facts["target_kind"], facts["directions"]
    words = WORDS[kind]
    headline, companion = facts["metrics"]["headline"], facts["metrics"]["companion"]
    held_out = f"held-out {naming.METRICS[headline]}"
    tables: dict[str, str] = {}

    tables["key"] = md.render(
        ["", "short", "what it is", "note"],
        [[row["kind"], row["short"], row["name"], row["note"]]
         for row in naming.key_table(facts["stage2"]["arms"], facts["target_kind"])])

    # ---------------------------------------------------------------------- the baseline
    base = facts["baseline"]
    budget = facts["stage3a"]["feature_budget"]
    if base is not None:
        tables["baseline"] = md.render(
            ["what it measures", "model", "numbers it uses", held_out,
             f"range over {base['n_seeds']} repeats", f"held-out {naming.METRICS[companion]}",
             f"within the budget of {budget}" if budget else "within the budget"],
            [[base["measures"], base["model"], base["n_features"], md.score(base[headline]),
              md.interval(base[f"{headline}_min"], base[f"{headline}_max"]),
              md.score(base[companion]),
              md.MISSING if base["within_budget"] is None else ("yes" if base["within_budget"] else "no")]])
    base_value = None if base is None else base[headline]

    # ---------------------------------------------------------------------- stage 2
    stage2 = facts["stage2"]["arms"]
    # A missing score sorts last whatever the metric's range, R² running below zero.
    order = sorted(stage2, key=lambda arm: (stage2[arm][headline] is None,
                                            -(stage2[arm][headline] or 0), arm))
    repeats = len(facts["stage2"]["seeds"])
    tables["stage2_all_arms"] = md.render(
        ["model", "feature source", held_out, f"range over {repeats} repeats",
         "shuffled labels", "gap", "features used"],
        [[*_model_source(arm), md.score(stage2[arm][headline]),
          md.interval(stage2[arm][f"{headline}_min"], stage2[arm][f"{headline}_max"]),
          md.score(stage2[arm][f"shuffled_{headline}"]), md.signed(stage2[arm]["paired_gap"]),
          md.count(stage2[arm]["n_features_used"])] for arm in order])

    rows, seen = [], set()
    for slot in headline_order(kind):
        arm = facts["headline_set_stage2"].get(slot)
        if not arm or arm in seen:
            continue
        seen.add(arm)
        entry = stage2[arm]
        clears = md.MISSING
        if base_value is not None and entry[headline] is not None:
            margin = entry[headline] - base_value
            clears = f"yes, by {margin:.3f}" if margin > 0 else f"no, by {-margin:.3f}"
        model, source = _model_source(arm)
        rows.append([model + ("" if entry["interpretable"] else " (reference)"), source,
                     md.score(entry[headline]), md.score(entry[f"shuffled_{headline}"]), clears,
                     md.count(entry["n_features_used"])])
    tables["stage2_headline"] = md.render(
        ["model", "feature source", held_out, "shuffled labels",
         "clears the base composition", "features used"], rows)

    # The judge's column only where the run's config ran the judge (`components/61_stage2_judge.md`
    # explains it).
    judged = facts["stage2"]["judged"]
    tables["stage2_reproducibility"] = md.render(
        ["model", "feature source", "same features again", "ideas found in more than one repeat",
         "found by another feature source", *(["how meaningful the features look"] if judged else [])],
        [[*_model_source(arm), md.frequency(stage2[arm]["reselection_kappa"]),
          f"{stage2[arm]['concepts_recurring']:.0f} of {stage2[arm]['concepts']:.0f}"
          if stage2[arm]["concepts"] and stage2[arm]["concepts_recurring"] is not None
          else "not measured",
          md.frequency(stage2[arm]["cross_source_agreement"]),
          *([md.score(stage2[arm]["judge_mean"], 2)] if judged else [])] for arm in order])

    tables["stage2_cross_arm_concepts"] = md.render(
        ["idea", "configurations reaching it", "runs on real labels", "runs on shuffled labels",
         "features on real labels", "features on shuffled labels", "enrichment p"],
        [[f"`{row['concept']}`", row["real_arms"], row["real_seeds"], row["permuted_seeds"],
          row["real_features"], row["permuted_features"], md.q_value(row["p_enriched"])]
         for row in facts["stage2"]["cross_arm_concepts"]])

    # ---------------------------------------------------------------------- stage 3A
    stage3a = facts["stage3a"]["arms"]
    reduced = f"reduced_{headline}"
    order3 = sorted(stage3a, key=lambda arm: (stage3a[arm][reduced] is None,
                                              -(stage3a[arm][reduced] or 0), arm))
    tables["stage3a_all_arms"] = md.render(
        ["model", "feature source", "started from", held_out, "95% interval",
         f"held-out {naming.METRICS[companion]}", "all its features",
         f"change in {naming.METRICS[headline]}"],
        [[*_model_source(arm), stage3a[arm]["n_start"], md.score(stage3a[arm][reduced]),
          md.interval(stage3a[arm][f"{reduced}_low"], stage3a[arm][f"{reduced}_high"]),
          md.score(stage3a[arm][f"reduced_{companion}"]),
          md.score(stage3a[arm][f"full_{headline}"]),
          md.signed(stage3a[arm][reduced] - stage3a[arm][f"full_{headline}"])] for arm in order3])

    for arm, entry in stage3a.items():
        # An arm whose reduction started from no feature has no table of features.
        if not entry["features"]:
            continue
        # The direction is the one the arm's model gave the feature before it was cut, which is the
        # only one recorded for every arm. Where a reduced model is drawn, its figure carries the
        # direction after reduction, and the two can differ.
        tables[f"features_{naming.slug(arm)}"] = md.render(
            ["#", "feature", "what it computes", words["direction"], words["alone"]],
            [[feature["rank"], f"`{feature['name']}`", md.clip(feature["description"]),
              direction_word(feature["direction"], directions),
              md.signed(feature["marginal_effect"], 2)]
             for feature in entry["features"]])

    # Features kept by several arms, matched on exact name. This undercounts, because two arms can
    # hold one idea under two names. The concept tables match on what the features compute instead.
    keepers: dict[str, list[str]] = {}
    for arm, entry in stage3a.items():
        for feature in entry["features"]:
            keepers.setdefault(feature["name"], []).append(arm)
    shared = sorted(((name, arms) for name, arms in keepers.items() if len(arms) >= SHARED_AT_LEAST),
                    key=lambda pair: (-len(pair[1]), pair[0]))
    tables["stage3a_shared_features"] = md.render(
        ["feature", "configurations", "which"],
        [[f"`{name}`", len(arms), "; ".join(naming.label(arm) for arm in sorted(arms))]
         for name, arms in shared])

    # ---------------------------------------------------------------------- stage 3B
    stage3b = facts["stage3b"]["arms"]
    by_both = sorted(stage3b, key=lambda a: (-stage3b[a]["reported_summary"]["stable_and_associated"], a))
    tables["stage3b_summary"] = md.render(
        ["model", "feature source", "candidates it chose from", "chosen again",
         "chosen again, allowing a near-duplicate", "holds on held-out rows", "model needs it",
         "chosen again and holds"],
        [[*_model_source(arm), md.count(stage3b[arm]["pool_size"]),
          *(stage3b[arm]["reported_summary"][key] for key in
            ("stable_exact", "stable", "associated", "necessary", "stable_and_associated"))]
         for arm in by_both])

    association = facts["metrics"]["association"]
    cost, cost_header = words["cost"]
    for arm, entry in stage3b.items():
        # The cost of removing a feature sits beside the test on that removal, as the held-out
        # association sits beside its q-value. On a large held-out set every test is significant,
        # and the size of what is lost is then the only column that still ranks the features.
        tables[f"evidence_{naming.slug(arm)}"] = md.render(
            ["feature", "chosen again", "with a near-duplicate", words["held_out"], "q",
             cost_header, "needs it (p)"],
            [[f"`{row['feature']}`", md.frequency(row["pi_hat"]), md.frequency(row["cluster_pi_hat"]),
              words["held_out_format"](row[association]), md.q_value(row["q_bh"]),
              md.signed(row[cost], 4), md.q_value(row["p_holm"])] for row in entry["reported"]])

    by_associated = sorted(stage3b, key=lambda a: (-stage3b[a]["reported_summary"]["associated"], a))
    tables["stage3b_tier2"] = md.render(
        ["model", "feature source", "candidates tested", "called at q <= 0.05",
         "called under the strictest correction", "of the model's features"],
        [[*_model_source(arm), md.count(stage3b[arm]["pool_size"]),
          _called(stage3b[arm], "bh"), _called(stage3b[arm], "westfall_young"),
          stage3b[arm]["reported_summary"]["associated"]] for arm in by_associated])

    tables["stage3b_plug_in_fdr"] = md.render(
        ["model", "feature source", "raw cut-off", "features called", "expected under the null",
         "share of calls that are noise"],
        [[*_model_source(arm), row["alpha"], row["n_called"],
          md.score(row["expected_null_called"], 2), md.score(row["plug_in_fdr"])]
         for arm, entry in stage3b.items() for row in entry["plug_in_fdr"]])

    # Two tables: the evidence, which a reader scans, and what each idea computes and the other
    # names it was found under, which a reader looks up afterwards.
    multi = [concept for concept in concepts if len(concept.arms) > 1]
    tables["concepts"] = md.render(
        ["idea", "model classes reaching it", "configurations", words["concept_direction"],
         "chosen again", words["held_out"], "holds on held-out rows"],
        [[f"`{concept.label}`",
          ", ".join(naming.MODELS[family]["short"] for family in concept.model_families),
          len(concept.arms),
          direction_word({"positive": directions["rising"], "negative": directions["falling"]}
                         .get(concept.direction), directions),
          md.frequency(concept.best_stability),
          # The strongest association any member reaches, beside the best q-value. On a large
          # held-out set every q falls to the smallest value the permutations can reach, and the
          # association is then the only column that still orders the ideas.
          words["held_out_format"](concept.strongest_association),
          md.q_value(concept.best_association_q)]
         for concept in multi])
    tables["concepts_detail"] = md.render(
        ["idea", "what it computes", "also found as"],
        [[f"`{concept.label}`", md.clip(concept.description, 150),
          ", ".join(f"`{name}`" for name in concept.member_names if name != concept.label)
          or "nothing else"] for concept in multi])
    return tables


#: The table holding the drawn symbolic model's equation.
EQUATION_TABLE = "stage3a_pysr_equation"

#: About how many characters of an equation's visible text, as `_visible` counts them, fit across
#: the text block, with a small margin: the A4 render holds about 95. A line holds what the
#: equation's left side leaves of this, or all of it less an indent when the left side stands on a
#: line of its own (`_aligned_lines`).
EQUATION_WIDTH = 93

#: The visible width of the `\quad` that indents a line, which `_visible` does not count.
QUAD_WIDTH = 4


def equation(facts: dict[str, Any], internals: dict[str, dict]) -> dict[str, str]:
    """The drawn symbolic model's equation, typeset with one top-level term to a line, a product too
    long for a line continued on the next, a fraction too wide for a line written as a division on
    the line, and a table of the features it names, strongest first, with what each measures and its
    mean absolute Shapley value. Empty when the run draws no symbolic model, or when that model
    used no feature."""
    slot = "best_pysr"
    if slot not in internals or internals[slot]["kind"] == NO_FEATURES:
        return {}
    structure = internals[slot]
    arm = facts["headline_set"][slot]
    described = {row["name"]: row["description"] for row in facts["stage3a"]["arms"][arm]["features"]}
    if facts["target_kind"] == BINARY:
        left = f"\\text{{log-odds of {_latex_text(facts['classes']['positive'])}}}"
    else:
        left = f"\\text{{predicted {_latex_text(facts['target']['name'])}}}"
    lines = _aligned_lines(left, structure["latex_terms"])
    display = "$$\n\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}\n$$"
    rows = [[feature["name"], md.clip(described.get(feature["name"]) or md.MISSING),
             f"{feature['importance']:.3g}"] for feature in structure["features"]]
    table = md.render(["feature", "what it measures",
                       f"mean absolute Shapley value ({structure['scale']})"], rows)
    return {EQUATION_TABLE: f"{display}\n\n{table}"}


def _aligned_lines(left: str, latex_terms: list[str]) -> list[str]:
    """The rows of the equation's `aligned` block: one top-level term to a row, each wrapped to the
    width it has. The terms are aligned after the left side, unless that leaves them less than half
    the line. A left side that long (a target with a long name) then stands on a row of its own and
    the terms start below it, indented, since a term wider than the room beside it would run past
    the margin."""
    width = EQUATION_WIDTH - len(_visible(left)) - len(" = ")
    if width >= EQUATION_WIDTH // 2:
        terms = [line for term in latex_terms for line in _wrapped(term, width)]
        return [f"{left} ={{}} & {terms[0]}", *(f"& {term}" for term in terms[1:])]
    terms = [line for term in latex_terms for line in _wrapped(term, EQUATION_WIDTH - QUAD_WIDTH)]
    return [f"& {left} ={{}}", *(f"& \\quad {term}" for term in terms)]


def _latex_text(text: str) -> str:
    for char in "\\_%&#${}":
        text = text.replace(char, "\\" + char)
    return text


def _visible(latex: str) -> str:
    """Roughly what `latex` shows, to measure a line by: the text kept, `\\exp` and `\\log` as the
    names they print, other commands and braces dropped, and the invisible brackets that close and
    reopen a broken line left out."""
    latex = re.sub(r"\\(left|right)\.", "", latex)
    shown = re.sub(r"\\text\{((?:\\.|[^{}\\])*)\}", r"\1", latex).replace("\\_", "_")
    shown = re.sub(r"\\(exp|log)(?![A-Za-z])", r"\1", shown)
    return re.sub(r"[{}^]", "", re.sub(r"\\[A-Za-z]+", " ", shown))


def _wrapped(term: str, width: int) -> list[str]:
    """`term` as lines of the aligned equation, each within `width` visible characters where it can
    be. A fraction too wide for a line is first written as a division on the line
    (`_inline_wide_fractions`). A product is then broken before a multiplication or division sign
    outside every bracket and braced group, and each continuation is indented. A factor still longer
    than a line is broken inside its brackets (`_split_in_brackets`), and one with nowhere to break
    keeps a line of its own."""
    term = _inline_wide_fractions(term, width)
    pieces, depth, start = [], 0, 0
    for at, char in enumerate(term):
        if char == "{" or term.startswith("\\left(", at):
            depth += 1
        elif char == "}" or term.startswith("\\right)", at):
            depth -= 1
        elif depth == 0 and term.startswith((" \\cdot ", " / "), at):
            pieces.append(term[start:at])
            start = at + 1
    pieces.append(term[start:])
    lines = [pieces[0]]
    for piece in pieces[1:]:
        if len(_visible(f"{lines[-1]} {piece}")) <= width:
            lines[-1] = f"{lines[-1]} {piece}"
        else:
            lines.append(f"\\quad {piece}")
    return [part for line in lines for part in _split_in_brackets(line, width)]


#: How a fraction opens in an equation's LaTeX: the command, then its numerator's braced group.
FRACTION = "\\frac"


def _inline_wide_fractions(term: str, width: int) -> str:
    """`term` with every fraction whose numerator or denominator is wider than `width` written as a
    division on the line, the denominator bracketed where it is a product. A fraction is typeset as
    one unbreakable box as wide as the longer of its two parts, so one wider than a line runs off
    the page, while a division on the line can be broken wherever a product or a bracket can. A
    fraction that fits is left as it is."""
    at = term.find(FRACTION + "{")
    if at < 0:
        return term
    top, after_top = _braced(term, at + len(FRACTION))
    bottom, end = _braced(term, after_top)
    top, bottom = _inline_wide_fractions(top, width), _inline_wide_fractions(bottom, width)
    if max(len(_visible(top)), len(_visible(bottom))) <= width:
        written = f"{FRACTION}{{{top}}}{{{bottom}}}"
    else:
        if _breaks(bottom, depth=0):
            bottom = f"\\left({bottom}\\right)"
        written = f"{top} / {bottom}"
    return term[:at] + written + _inline_wide_fractions(term[end:], width)


def _braced(text: str, at: int) -> tuple[str, int]:
    """The contents of the braced group opening at `text[at]`, and the index after its closing
    brace. A brace escaped with a backslash, as inside a feature's name, opens and closes nothing."""
    if text[at] != "{":
        raise ValueError(f"no braced group opens at {at} of {text!r}.")
    depth, index = 0, at
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[at + 1:index], index + 1
        index += 1
    raise ValueError(f"the braced group opening at {at} of {text!r} never closes.")


#: Where a line may be broken, with the signs first: before a plus or minus sign, and before a
#: multiplication or division sign.
SIGNS, PRODUCTS = (" + ", " - "), (" \\cdot ", " / ")


def _breaks(line: str, depth: int | None = None) -> list[tuple[int, bool, int]]:
    """Every place `line` may be broken, as (how many brackets deep, whether it is a multiplication
    or division sign, where): before a sign outside every braced group, and not straight after a
    reopened bracket, where the line was broken before. `depth` keeps the places at that depth."""
    found, deep, braces = [], 0, 0
    for at in range(len(line)):
        if line.startswith("\\left", at):
            deep += 1
        elif line.startswith("\\right", at):
            deep -= 1
        elif line[at] == "{" and not line[:at].endswith("\\"):
            braces += 1
        elif line[at] == "}" and not line[:at].endswith("\\"):
            braces -= 1
        elif (not braces and line.startswith(SIGNS + PRODUCTS, at)
              and not line[:at].endswith(REOPENED)):
            found.append((deep, line.startswith(PRODUCTS, at), at))
    return [place for place in found if depth is None or place[0] == depth]


#: What opens a line continued inside a bracket: the reopened bracket, then an empty group.
REOPENED = "\\left. {}"


def _split_in_brackets(line: str, width: int) -> list[str]:
    """`line` broken inside its brackets while it is wider than `width`. The break is made in the
    shallowest bracket where the part before it then fits the line, before a plus or minus sign
    where that bracket has one that serves, and otherwise before a multiplication or division sign,
    and as late in the line as fits. Where no break leaves a part that fits, the one taken is the
    first plus or minus sign in the shallowest bracket, or the first multiplication or division sign
    there where it has neither. Both parts are broken again. LaTeX pairs each `\\left` with a
    `\\right` on the same line, so every bracket open at the break is closed with an invisible
    `\\right.` and opened again on the next line with an invisible `\\left.`, followed by an empty
    group that keeps the sign a binary operator, spaced as the other lines' signs are."""
    if len(_visible(line)) <= width:
        return [line]
    places = [place for place in _breaks(line) if place[0] > 0]
    if not places:
        return [line]
    fitting = [place for place in places if len(_visible(line[:place[2]])) <= width]
    if fitting:
        depth, _, cut = min(fitting, key=lambda place: (place[0], place[1], -place[2]))
    else:
        depth, _, cut = min(places)
    head = line[:cut] + " \\right." * depth
    tail = "\\quad " + "\\left. " * depth + "{}" + line[cut:]
    return [*_split_in_brackets(head, width), *_split_in_brackets(tail, width)]


def _called(entry: dict[str, Any], method: str):
    """How many of the pool a correction called at 0.05. A JSON round trip turns the key into the
    string "0.05", so both forms are tried."""
    called = entry["pool_called"].get("0.05") or entry["pool_called"].get(0.05) or {}
    return called.get(method, md.MISSING)
