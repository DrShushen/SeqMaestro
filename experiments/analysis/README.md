# Analysis: read a finished run back

Each of these scripts works on one run. It writes its tables into that run under names of its own,
and changes nothing a stage wrote. The rules every stage follows are in
[`../README.md`](../README.md). This file says what each script answers and when to use it.

**None of them reads the test split.** The analyses in the standard pipeline work from what each
cell reported and from its values on validation, and they run before the reveal. A script that
reports performance reads test scores only after the run has been revealed, and says which split its
numbers come from (see [`../../docs/heldout.md`](../../docs/heldout.md)).

**How each number is computed is documented in [`../../docs/analyses/`](../../docs/analyses/)**,
one maintained page per analysis, with the reasoning, the worked arithmetic and the failure modes.
When you change an analysis, change its page in the same commit:

| script | page |
|---|---|
| `analyse_stability.py`, `analyse_co_discovery.py` | [`stability_and_co_discovery.md`](../../docs/analyses/stability_and_co_discovery.md) |
| `analyse_recurrence.py` | [`recurrence.md`](../../docs/analyses/recurrence.md) |
| `run_judge.py` | [`judge.md`](../../docs/analyses/judge.md) |
| `direct_effect_test.py`, `feature_group_dependence.py` | [`direct_measurement.md`](../../docs/analyses/direct_measurement.md) |
| anything reading `direction` or `marginal_effect` | [`direction.md`](../../docs/analyses/direction.md) |
| what every one of them reads | [`feature_record.md`](../../docs/feature_record.md) |

## What each one answers

**In the standard pipeline**, run by `../runners/run_experiment.py` in this order, each with
`--run-dir <run>`:

| | question | writes |
|---|---|---|
| `analyse_stability.py` | does an arm *select* the same features again across seeds? | `results/stability.csv` |
| `analyse_co_discovery.py` | did more than one arm find the same thing? | `results/co_discovery.csv`, `results/co_discovery_by_arm.csv` |
| `analyse_recurrence.py` | does an arm on a written source find the same features again, arrive at the same ideas again, and do different such arms arrive at the same idea? | `results/feature_recurrence.csv`, `results/concept_recurrence.csv` with its record `concept_recurrence.json`, `results/cross_arm_recurrence.csv` |
| `run_judge.py`, where the run's config has `grid.judge: true` | how biologically significant would each feature be, if true? | `judge/judge_summary.json`, `judge/judge_scores.json`, `judge/panel_key.json` |

The last two ask a language model, so each keeps what is on file while the features, the settings
and the description it was computed from are unchanged, and `--force` asks again
(`seqmaestro/pipeline/analyses/inputs.py`). How many features of a cell an analysis takes, how
often the judge rates each one and how the recurrence analysis groups are settings of the run, in
the `analyses` section of its config (`../configs/README.md`), so an analysis started by hand
takes them as the driver's does.

**On demand**, run by hand, because each needs an argument only the caller can choose or runs in a
different environment. `--help` lists each one's arguments:

| | question | takes |
|---|---|---|
| `direct_effect_test.py` | does *this* feature differ between the classes, by how much, and could it be chance? | `--dataset` (a table of the dataset), `--label-column`, and `--features` (cells' `features.json`) or `--columns` |
| `export_features_xlsx.py` | what did every arm report, as a spreadsheet to hand to a collaborator? Performance is on the test split once the run is revealed, on validation before. Runs under `SeqMaestro_LocalUtils` | `--run-dir`. Writes `<run>/features.xlsx` unless `--out` names another place |
| `feature_group_dependence.py` | how much of each feature *is* the thing the split groups on? | `--dataset`, `--sequence-column`, `--group-column`, and `--pools` with how k-mers are counted (`--k-range`, `--kmer-normalise`, as the run's config states them) or `--features`, each file listed under its path |

**The report**, after the reveal, in this order (`../../process/reporting.md` gives the whole
sequence, and `../../docs/report/README.md` says what the build decides):

| | environment | writes, under `<run>/report/` |
|---|---|---|
| `build_report_inputs.py` | Core | `facts.json`, `tables/`, `internals/`, `figure_inputs.json`, read from the run as its config lists it |
| `build_report_figures.py` | LocalUtils | `figures/`, `figures.json`, from `figure_inputs.json` alone |
| `assemble_report.py` | either | `REPORT.md`, from the `REPORT.src.md` a writer wrote against `../../docs/report/TEMPLATE.md` |
| `render_pdf.sh` (with `page_layout.lua`) | none; `pandoc` and `xelatex` | `REPORT.pdf`. `REPORT_ORIENTATION=landscape` in its environment renders the whole report landscape |

Run `feature_group_dependence.py` **before** a grid. It says whether a grouped split isolates what it
promises, and the answer is per pool: one split can isolate the grouped variable from one
pool and not from another. What a design can detect at all is part of the dataset, measured
when Stage 1 finalises it and recorded in its `dataset.json` (see
[`../../docs/dataset.md`](../../docs/dataset.md)). `../prepare/design_power.py` answers the same
question for a split still being planned.

Run `direct_effect_test.py` when a collaborator asks about one feature instead of about the model.
It compiles features from the code a cell recorded in its `features.json`, so the test runs on the
feature the arm reported. Its statistics come from `seqmaestro.harness.effect`.

## Stability, co-discovery and recurrence are three different questions

All three cluster features and report agreement, so they are easy to confuse. They differ in the
axis they compare along and in what counts as the same feature.

| | axis | "the same feature" means | needs |
|---|---|---|---|
| `analyse_stability.py` | seeds, within an arm | the same name, or values that correlate | features evaluated on a shared reference split |
| `analyse_co_discovery.py` | **arms, within a seed** | values that correlate on the reference split | every arm evaluated on the same rows |
| `analyse_recurrence.py` | **seeds within an arm, and arms against each other** | identical text, or a language model says they measure the same property | nothing shared; names and descriptions only |

Co-discovery and recurrence are the easiest pair to confuse, because both produce something called
a concept. Co-discovery asks *"did anything else find this?"* and answers it behaviourally, by
correlating values, because features from different arms have nothing in common except their values.
Recurrence asks *"did this arm think of it again?"*, and its concept view cannot be answered
behaviourally: a generative arm re-expresses one idea over a shifted window every seed, and two
shifted windows do not correlate. For the same reason recurrence complements `top_n_kappa` (from
`analyse_stability.py`) and does not replace it. `top_n_kappa` measures *selection* and the concept
view measures *proposal*, and an arm can be stable in one and not the other.

## Each of the three needs its null read first

Stability, co-discovery and recurrence each report a permuted-label column beside the real one,
and in several of them the raw figure points the wrong way until it is compared with the null.
(The judge rates real-label features only, unless the run's config has `analyses.judge.labels`
hold `permuted` too.) Arms that share a pool
agree structurally under any labels. A language-model proposer reads the task description, which
permuting the labels does not change, so it proposes much the same repertoire under both. A number
quoted without its null is not a result.

Where the null rests on few independent draws, say how many before quoting a gap against it. See
`grid.permutation_scope` in the run config, which decides whether a run has one draw per seed or one
per cell.
