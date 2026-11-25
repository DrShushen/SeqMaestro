---
name: reporting
stage: report
---

# The report

The run is finished: the held-out split has been opened once, and every stage's numbers are on
disk. This stage turns them into **one document** that covers all four stages, for a
bioinformatician and a biologist reading the same page.

Read [`README.md`](README.md) in this folder first if you have not. This stage begins when the
reveal is complete and the reliability panel's tables exist
([`reliability-panel.md`](reliability-panel.md), "Check after the reveal").

## What this stage must leave behind

```
experiments/runs/<date>_<name>_full/report/
    facts.json, tables/, internals/, figure_inputs.json    built from the run (Core)
    figures/, figures.json                                  drawn from figure_inputs.json (LocalUtils)
    REPORT.src.md                                           what you write
    REPORT.md, REPORT.pdf                                   expanded, and rendered
```

## Read these first, in this order

1. [`../docs/developer/report.md`](../docs/developer/report.md): what the report layer decides and why
   (the headline set, the reference line, concepts, the checks behind every drawn model).
2. [`report-template.md`](report-template.md): the section list, what each section must contain,
   the style rules, and the checks before handing over. **Follow it.**
3. **A worked example** if one exists: a finished report of an earlier run, read beside the template
   to see what "good" looks like. Ask the user which one to use. Do not copy its findings, its
   wording about its dataset, or its numbers.
4. **The dataset's `README.md`** (`data/<dataset>/README.md`) and its `dataset.json`. Stage 1 of the
   report is a rewrite of these for the reader.
5. **The task description the language models were told**: `experiments/datasets/<task>.json`,
   where `<task>` is `dataset.task` in the run's `config.yaml`. On a blinded run its `judge_only`
   holds the fact that was withheld from the models that wrote features. The config's
   `description` is a note on why the variant exists, which no model was told. Any guidance the
   models were given on the form of the features is the config's
   `grid.generated_features.guidance`, which `facts.json` holds as `stage1.feature_guidance`.

## Record who you are

Before you write, record yourself for the report (`README.md`, "Rules that hold at every stage"):

```bash
RUN=experiments/runs/<date>_<name>_full
$CORE experiments/runners/record_agent.py --run-dir $RUN --agent "<the agent you are>" \
    --model "<the model you run on>" --stages report
```

The report names every agent the run records, with the model it runs on, in the table of agents
(`tables/agents.md`), beside the table of the language model the run called
(`tables/language_models.md`). `assemble_report.py` renders the table of agents again from the
run's record before it fills the placeholders, so the table names you even though the inputs were
built before you recorded yourself. It warns when no agent has recorded itself for the report.

## Build the inputs

The driver builds them after the reveal, so on a run it finished they are already there. To build
them again (after a change to the report layer, for example), run two commands in two environments
from the repository root (`ENVIRONMENTS.md` has the interpreters):

```bash
RUN=experiments/runs/<date>_<name>_full
$CORE experiments/analysis/build_report_inputs.py --run-dir $RUN            # facts, tables, internals
$LOCALUTILS experiments/analysis/build_report_figures.py --report-dir $RUN/report   # the figures
```

The first reads the run as its config describes it, and refuses a run whose reveal is not
complete. A DEFT tree in the headline set is loaded in DEFT's environment, and an IGSR model's
terms are evaluated in IGSR's, so the first command needs those interpreters where the run has
such arms. The second draws only from what the first wrote. Then write
`$RUN/report/REPORT.src.md`, and run:

```bash
$CORE experiments/analysis/assemble_report.py --report-dir $RUN/report
bash experiments/analysis/render_pdf.sh $RUN/report/REPORT.md
$CORE experiments/analysis/build_status_page.py --run-dir $RUN
```

The last builds the run's status page again, so that it links to the report.

**The class names come from the dataset** (`dataset.json`), with their meanings. They are the two
words beside every direction in the report. If the dataset's record leaves in doubt which way round
they run, say so first among the open questions. **A continuous target is named and explained by
the dataset too** (`target.name` and `target.meaning`, in `facts.json`'s `target`): every direction
in its report says whether higher values of a feature raise or lower it, so if the record leaves in
doubt which way its sign runs, that too comes first among the open questions.

## Write the report

Write `REPORT.src.md` against the template. It is markdown with four kinds of placeholder, which
`assemble_report.py` fills in:

| placeholder | what it becomes |
|---|---|
| `{{component:50_reliability_overview}}` | a run-independent explanation from `seqmaestro/report/components/` |
| `{{table:stage3b_summary}}` | a rendered table from `report/tables/` |
| `{{figure:stage3a_tree}}` | a figure, with the caption and width the build recorded |
| `{{param:half_samples}}` | a setting the run recorded, from `facts.json`'s `parameters` |

- **Never retype a number.** Include the table and say what it means. A number quoted in prose comes
  from `facts.json`.
- **A placeholder is substituted verbatim**, so do not start a sentence with one: it would need a
  capital and would not get one.
- **Components have their own headings**, so do not write one above them. Three have none: the
  glossary, and `33_model_reference` and `61_stage2_judge`, each of which continues the component
  included before it.
- **Say nothing about the page.** No image widths, no page breaks, no LaTeX. The render rotates the
  figures the build measured as too wide and starts each appendix on a fresh page.
- **Parameters, not constants.** Where your prose needs a number the run chose, use `{{param:...}}`.
- **A symbolic model's equation is a table**, `{{table:stage3a_pysr_equation}}`: the equation
  typeset one term to a line, with the features' names as text, and a table of what each of its
  features measures. It goes after the model's component and before its two figures, as the
  template says. Describe the equation in words (which features it adds, multiplies, divides or
  puts inside a logarithm), and never retype it.

## The angle

The pipeline delivers two different things, and both are results.

1. **A model** you could apply to a new sequence, small enough to interpret end to end.
2. **A set of ideas** that held up across independent arms and on sequences nothing was fitted
   on, which is what a follow-up is designed around.

These are different goals, so one feature need not serve both. Write the report that way. **Do not
write the report as a model followed by a list of reasons to distrust it.** Where the reliability
panel is silent about a feature, say in a line what it is silent about and why, and move on. Where
the study lacked the power to resolve something, state that once, as a fact about its size.

**Keep fact and interpretation apart.** Everything outside a marked block is a number or a statement
of what was done. Judgement goes inside a block headed `> **AI Agent Analysis Note**`, and every
stage section ends with one.

## Where everything is

Every number a report quotes was written into the run by one stage. `facts.json` names the files
behind each configuration's numbers under `files`. That is the map a reader asking "where does this
come from" needs, and Appendix D of the report is written from it.

| what | where in the run | written by |
|---|---|---|
| the dataset: the target (classes, or a continuous target's name and meaning), split, preparation steps, checks, detection floor | `dataset.json` beside the splits (the config's `dataset.splits`) | Stage 1, `experiments/prepare/finalise_dataset.py` |
| the composition baseline, per seed | `baseline/seed_<n>/metrics.json` | `run_baseline.py`; test block by the reveal |
| every arm's cells, per labels and seed | `results/<model>/<pool>/<labels>/seed_<n>/`: `metrics.json`, `features.json`, `run.json` | Stage 2 (`run_arm.py`); test blocks by the reveal |
| whether an arm selects the same features again | `results/stability.csv` | `analyse_stability.py` |
| what other arms found too | `results/co_discovery.csv`, `co_discovery_by_arm.csv` | `analyse_co_discovery.py` |
| ideas found again, and across arms | `results/concept_recurrence.{csv,json}`, `cross_arm_recurrence.csv`, `feature_recurrence.csv` | `analyse_recurrence.py` |
| how meaningful each feature looks, where the config runs the judge | `judge/judge_summary.json` | `run_judge.py` |
| each reduced arm: its record, steps and three models | `reduction/<model>/<pool>/`: `reduction.json`, `steps.npz`, `full/`, `reduced/`, `top_n/` | `run_reduction.py`; test blocks by the reveal |
| each profiled arm's panel | `reliability/<model>/<pool>/`: `reliability.json`, `tier1.json`, `tier2_*.json`, `tier3.json`, `checks.json`, `profile_*.csv` | `run_reliability.py`, before and after the reveal; Tier 2 on the test split by the reveal |
| one row per profiled arm | `reliability/summary.csv` | `run_reliability.py --after-reveal` |
| what the reveal sealed, and when | `reveal.json` | `run_reveal.py` |
| the agents that carried out the run, and the model each runs on | `agents.json` | each agent, with `record_agent.py` |
| the language model each part of the run called, and how | `language_model` in a cell's `run.json` (a proposer's under `featuriser`), in `judge/judge_summary.json` and in `results/concept_recurrence.json` | the part that made the calls |

What each of those computes is in [`../docs/`](../docs/developer/index.md): `dataset.md`, `suite.md`,
`baseline.md`, `heldout.md`, `reduction.md`, `reliability.md`, and `analyses/` for the Stage 2
analyses.

## Check before handing over

- Every number traced to a table or figure you included. Nothing retyped, nothing recomputed.
- Every feature name copied, never typed.
- Every reported feature has a direction, or a stated reason it has none.
- A symbolic model whose cells count rows its equation was undefined on (`filled_rows` in
  `metrics.json`) says so beside its held-out numbers.
- For a binary target, the class balance is stated, and the metrics quoted are the ones that
  balance allows. For a continuous target, the target's spread is stated, and wherever a few
  extreme values could drive R², Spearman's rho is quoted beside it.
- The reference line is named in Stage 2 and Stage 3A, with whether it fits the reduction's budget.
- Every analysis has its explanation before its numbers.
- No repository vocabulary, no directory names, no discussion of our tooling, costs or environments.
- `assemble_report.py` reports no unused tables or figures that should have been included, and
  no warning that the run records no agent for the report.
- **The PDF builds and you looked at it.** Render its pages to images and look at them for tables
  running past the right margin, figures too small to read and headings orphaned at the foot of a
  page.

Then report back to the user.

## If something does not fit

The build raises an error rather than falling back to a default, and each message says what to add.

- **A model class or feature source it does not know** needs an entry in
  `seqmaestro/report/naming.py`, which decides how a reader sees it.
- **An arm missing from Stage 3B** is expected: the panel leaves out black-box models, whose fit
  cannot be interpreted feature by feature, and a config can leave an arm out of the reduction or the
  panel with a reason (`not_reduced`, `not_profiled`). `facts.json` records each under
  `stage3b.not_profiled`. Say how many were profiled and why the rest were not.
- **A Stage 2 analysis the run did not produce** (recurrence and the judge need a language model) is
  reported in the tables as not measured, and the build lists what is absent. A run whose config leaves
  the judge out (`grid.judge`) has no judge column, and the judge's component is left out with it.
- **Something the layer gets wrong for this dataset**, such as an explanation that only fits
  another dataset or a figure whose legend covers the data, is fixed in the layer and not worked
  around in the report. Follow the two rules in [`../docs/developer/report.md`](../docs/developer/report.md)
  ("Changing the layer"). Expect a new dataset to need at least one such change.

## This file is a floor, not a ceiling

It lists what the report must contain. It does not list everything worth saying. If the numbers show
something the template has no section for (an arm that behaves unlike the rest, a check that
nearly failed, a result that is too good), say it in the section it belongs to, inside an Analysis
Note if it is judgement, and raise anything that questions the design with the user.

## Do not

- Do not compute a new number, or reopen the held-out split for one. If a number you want is
  missing, say so or leave it out.
- Do not edit the run's files outside `report/`.
- Do not copy a finding into `docs/` or the components. A finding about this run belongs in this
  run's report.
