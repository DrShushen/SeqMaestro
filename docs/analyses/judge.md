# The significance judge

> **Maintained page.** Change it in the same commit as
> `seqmaestro/pipeline/analyses/judge.py`, which `run_judge.py` calls, or anything under
> `seqmaestro/judge/`. The prompt text in particular: changing it changes what past scores mean.

`run_judge.py` asks a language model, one blinded feature at a time:

> **How biologically significant or interesting would this feature be, if it were true?**

It is the only axis in this system that measures *worth* and not *reproducibility*. Everything else
asks whether a feature comes back. This asks whether anyone should care.

---

## 1. The question, and the four words doing the work

**"If it were true" is load-bearing.** It separates significance from correctness, so the judge is
not asked to decide truth from a description it cannot verify. It cannot do that, and would do it
badly while sounding confident.

It also sharpens the negative control. A noise-derived feature *can* be interesting if true, and the
question allows that. So permuted features scoring as highly as real ones is informative: it means
the score tracks **how a feature is written** and not what produced it.

**One question, not a rubric.** A rubric of parts (mechanistic plausibility, specificity, standing
relative to known biology, testability) would score things that mostly co-vary. Averaging them gives
a number nobody can interpret, and reporting them separately quadruples the output without changing
any decision.

**The scale is anchored**, because an unanchored 1 to 5 scale drifts between calls and cannot be
compared across arms:

| | |
|---|---|
| 1 | Trivial or uninformative. Tells a biologist nothing they would act on. |
| 2 | Marginal. A weak or very generic observation. |
| 3 | Moderately interesting. Plausible and worth noting, but not surprising. |
| 4 | Clearly interesting. Would prompt a biologist to look further. |
| 5 | Highly significant. Would change how someone thinks about the mechanism. |

The system prompt tells the judge to be discriminating, in those words: *most candidates are
ordinary, and a scale on which everything scores 4 measures nothing.*

---

## 2. Building the panel

Three jobs, all cheap and all easy to get subtly wrong, which is why they live in
`seqmaestro/judge/panel.py` and not inline in the script.

**Selection.** The top *N* by importance of each real-label cell (`analyses.top_n` in the run's config, 20 in the shared one), so an arm puts up
to N features on the panel per seed, and a feature it uses at every seed is rated once for each.
Judging every feature is unaffordable, and the tail of a vocabulary running to thousands is not
what any arm claims to have discovered.

**Blinding.** Each entry gets an opaque identifier, such as `F0007`. The arm, the seed and the label
condition travel *beside* the panel, never inside it, so a prompt cannot mention them. This holds by
construction and does not depend on anyone remembering. `tests/test_judge.py` asserts that no arm
name reaches a prompt.

**Ordering.** Shuffled with a seed, so an arm's features are not adjacent and a drifting judge
cannot systematically favour whoever went first. **Identifiers are assigned after shuffling.**
Sequential ids assigned before shuffling would put each arm in a contiguous block of ids, which
leaks the arm through the id itself.

**Only real-label features are on the panel by default, and in every run the driver makes.**
A config whose `analyses.judge.labels` holds `permuted` too adds the permuted-label features. They then enter as ordinary
entries, indistinguishable to the judge, and are separated again only when the scores are
summarised. Section 5 says what such a control can show, and why it is not judged by default.

`panel_key.json` says what each blinded id stands for: the arm, the label condition and the seed
the feature came from, its name and its importance. The judge is shown an item under its id and
none of this. `judge_scores.json` names each item's arm and label condition too, since the summary
is taken by arm, so neither file is blind: the blinding is of the judge.

---

## 3. What the judge is shown

For each item:

```
<the dataset description, including anything withheld from the models>

How biologically significant or interesting would this feature be, if it were true?

Feature name: <name>
Description: <description, or "(none given)">
Stated rationale: <rationale>            # when the record carries one
Observed in the data: this feature takes higher values in the <positive|negative> class.
                                         # only where the run's grid has judge_direction on. For a
                                         # continuous target the line says whether higher values
                                         # go with a higher or a lower target
Implementation: <source>                 # only where the config has analyses.judge.show_code

Scale: ...
Answer with JSON only: {"score": <1-5>, "reason": "<one sentence>"}
```

### Two senses of blinded meet here, and neither implies the other

The **feature** is blinded: the judge is not told which arm produced it or how it scored.

The **dataset** may be blinded in the other direction. `TaskContext` carries a `judge_only` field:
one fact told to the judge and withheld from every model consumer. The proposer and the generative
methods are what the experiment tests, so a fact they could *retrieve* instead of discovering (the
name of the protein whose loss defines the classes, say) makes the result unreadable if they are
handed it. The judge is not under test. It is the instrument, and it cannot say how significant a
feature would be without knowing what the phenotype is.

The plain rendering is the blind one: a caller reaching for `render()` gets what the models get, so
a caller who forgets which one it wanted under-discloses and leaks nothing.

### Whether it runs

A run's config says, in `grid.judge`, and `_shared.yaml` sets it false: the judge asks a
language model about the 20 most important features of each real-label cell, three times each
(`analyses.top_n` and `analyses.judge.repeats`), so its calls grow with the arms and the seeds, and nothing after Stage 2 depends on
its scores. A dataset's config sets `judge: true` to run it. A run without it has no judge stage,
and its report has no judge column.

A real run needs the credentials. `--dry-run` builds the panel, prints the first prompt and makes
no call. The judge writes `judge_summary.json`, `judge_scores.json` and `panel_key.json` under
`<run>/judge/`, and keeps them when it is started again while the features, the settings and the
description they were computed from are unchanged. `--force` rates again.

### The direction line

A setting of the run, `grid.judge_direction`, which `_shared.yaml` sets true. The judge takes it
from the run's recorded config, however it is started, so every rating of a run was made under one
setting.

It changes what is being asked. "How significant would this be if true" is barely answerable for a
count of upstream start codons, which is plausible either way round. Told which way it runs, the
judge is scoring a **specific claim** and can mark down one that contradicts what is known. That is a
better instrument and a *different* one, so **scores from a run with it on cannot be read beside
scores from a run with it off**, and the flag is recorded in `judge_summary.json` for that reason.

**The sign, never the size.** An effect size would let the judge separate real features from the
permuted-label controls that every arm's mean is read against. A sign is a coin flip for a control
feature, so it discriminates nothing. That is also why the direction shown is the **marginal** one
and not the model's: it is a statement about the data, which is what the question asks about, and
it is always one of two words, where the model's direction admits a third.

The run also records `features_with_direction`, because asking for directions and getting none is
otherwise silent. A feature with no `marginal_effect` goes out without a direction line, and the
setting would read as on while its prompts carried none.

---

## 4. Scoring and aggregation

**Every feature is judged `analyses.judge.repeats` times, in independent calls**, so the spread across repeats
measures how reproducible the judgement was.

**A refusal is not a score.** A call that fails, times out or returns something unparseable yields
`None`, and `None` is carried through to the summary and counted as `n_unanswered`. Treating a
failure as a middling score would quietly pull every arm towards the mean, and the arms whose
features are hardest to parse would be flattered most. `analyses.judge.min_answered` fails the stage when too
much of the panel went unanswered, because a mostly unanswered panel is a failed stage and not a
result.

Reply parsing is tolerant within limits: the JSON object asked for, wherever it sits in the reply
and inside a fenced block or not, or a reply that is the score alone. Anything else is unanswered.
Models wrap JSON in prose often enough that insisting on clean JSON would discard good answers,
and inventing a score from a reply containing none would be worse. A number picked out of a
sentence is not taken, since a reply that restates the scale would be scored by the scale's first
number.

### The summary

Per (arm, label condition): `n_features`, `mean`, `median`, `mean_spread_across_repeats`,
`n_unanswered`, and `control_gap` when permuted-label items were judged.

**The noise floor, where permuted-label items were judged.** The mean over every permuted-label
item is the level this instrument returns when there is nothing to find, and each arm's gap above
it is `control_gap`. An arm scoring 3.6 against a floor of 3.5 has said nothing, while the same 3.6
against a floor of 2.4 is a result. A run judged on real-label features alone has no floor and no
`control_gap`, and its means compare arms with each other.

---

## 5. What it cannot support

**It scores writing as much as biology.** A feature with a fluent description and a plausible
rationale scores well whether or not it came from real labels. Judging the permuted-label features
measures that for an arm that selects from a fixed vocabulary, because its permuted features are
different features: if they score as highly as the real ones, the judge is scoring how a feature
is written.

**The permuted-label features are no control for a language-model proposer, so they are not judged
by default.** The proposer takes its features from the task description and its own knowledge, and
the labelled rows are a small part of what it is shown. On shuffled labels it proposes nearly the
same features, so the judge would score the same panel twice, and the number would look like a
control without being one. `concept_recurrence.csv` shows this from the other side, by counting how
many ideas an arm repeats under each label condition, for every arm on a written source. A single
arm's mean therefore describes its features and is no evidence about the labels. The control this
axis needs is a shuffled task description, which is not built.

**It is not evidence the feature is true.** The question sets truth aside. Pair a high judge score
with the direct-effect test ([`direct_measurement.md`](direct_measurement.md)) and with recurrence
before treating a feature as a finding.

**The judge sees the dataset description, and that description is factual and silent about known
confounds.** Telling it that GC content is confounded here would have it mark GC features down,
which is a judgement about whether a feature is *real*, the very thing "if it were true" sets aside.
Such warnings belong in the report.

**Prompt changes invalidate comparisons.** The rubric text, the scale wording, `analyses.judge.show_code` and
`grid.judge_direction` all change what the number means. Any of them moving is a reason to re-run
every arm being compared, not just the new one.
