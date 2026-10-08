# The report

**What it produces.** One document per run, for a bioinformatician and a biologist reading the same
page, covering all four stages: how the data was prepared, which representation and model class
carry the signal, the few-feature models worth showing, and which of their features would survive a
repeat of the study.

**Nothing in the report layer computes a result.** Every number comes from a stage that already
produced it, and the test split's numbers are the ones the reveal scored. The layer reads, renames,
tabulates, draws and assembles. The code is in `seqmaestro/report/`. How to build a report, and what
the writer does, is in [`process/reporting.md`](../../process/reporting.md).

| here | |
|---|---|
| [`TEMPLATE.md`](TEMPLATE.md) | the report template: the section list, what each section must carry, the style rules, and the checks before handing over. What a writer, human or model, is handed |
| [`components/`](components/) | run-independent explanations, included verbatim: how an additive model works and how to read its figure, what each reliability tier measures, the glossary |

---

## Four separations, and why each is where it is

**Numbers from prose.** The build writes `facts.json` and `tables/*.md`. A report source is prose
with placeholders, and assembling it fills them in: `{{component:NAME}}` for an explanation,
`{{table:NAME}}` for a rendered table, `{{figure:NAME}}` for a figure with the caption and width the
build recorded, and `{{param:NAME}}` for a setting the run chose. A report includes a table and says
what it means, and never retypes a number, so the two cannot drift. An unresolved placeholder stops
the assembly, so it never reaches the reader as braces.

**Run-independent explanation from run-specific finding.** A component says what a q-value is, and
the report says what this run's q-values were. Improving an explanation improves every report, and
no explanation is rewritten per dataset. A number a component needs that differs between runs (how
much resampling the panel did, how many features the reduction kept) is a parameter, never a
constant. **Where the kind of target changes what a component explains** (the metric, the classes,
the loss a refit is compared in), the component has a twin per kind, `NAME.binary.md` and
`NAME.continuous.md`, and the run's kind (`facts.json`'s `target_kind`) chooses the twin when the
report is assembled. A component never switches on the kind inside its prose: a reader of one kind
of report meets nothing written for the other. A component that serves both kinds is one file,
`NAME.md`, and says nothing either kind would find wrong.

**The document from the page it is printed on.** The report is plain markdown and says nothing about
page breaks or orientation, and a figure carries only the share of the text width the build
recorded for it. `render_pdf.sh` renders it portrait on A4, and a pandoc filter
gives a rotated page to any figure the figure build measured as too wide for a portrait column.

**Choosing what to show from deciding what it means.** Which configurations lead the body is a rule
(below), applied identically to every run, so a reader can see what the rule was. What the result
means is the writer's judgement, marked as such in the document.

## What the layer decides

**The headline set.** The best configuration of each interpretable model class (additive, sparse
linear, tree, symbolic, and IGSR on a continuous target), the best configuration built on features
enumerated in advance, and the strongest black-box reference. Chosen twice: on Stage 2's mean held-out headline (AUROC for a binary target,
R² for a continuous one), and on the Stage 3A reduced model's held-out headline, because the two
stages answer different questions and an arm can lead one and not the other. The sparse linear
class is the logistic model for a binary target and the least-squares one for a continuous one.
Ties go to the arm whose name sorts first. Whether a model class is interpretable is declared in
`seqmaestro/arms.py` (`black_box`), which takes it from the `interpretable` attribute of each
model fitted in Core and states it for DEFT's tree and for IGSR. A tree is read node by node, so
it counts as interpretable.

**The reference line.** The composition baseline ([`../baseline.md`](../baseline.md)): its mean
held-out headline over the grid's seeds, with its range and whether its feature count fits the
reduction's budget.

**The kind's names.** `facts.json` names a score's key by the metric of the run's kind (`auroc` and
`auprc`, or `r2` and `spearman_rho`), and its `metrics` block says which, with the chance level, the
panel's association and its loss. Its `directions` block says how each direction word is printed:
as the class a binary target's feature argues for, or as which way a continuous target moves.

**Concepts.** The reduced models' features, grouped across arms by what they compute and not by what
they are called. Every column the reduction kept for a profiled arm, one its reduced model gives
no weight included, is evaluated on D (train + validation, never the test split). Our arms are
evaluated through the reduced cell's saved featuriser, and DEFT's and IGSR's are taken from the
panel's evaluation over D, made on each method's side. The columns are then grouped by the harness's
behavioural matcher at its own threshold, with the grouping chained transitively. Each concept
carries the arms and feature sources that reached it, a direction where its members' separations
agree in sign, and the strongest evidence any member has from the panel. Concepts are ordered by how
many arms reached them, then by the best Tier 2 q-value, then by the best Tier 1 selection
frequency. Two limits apply: transitive chaining can merge two ideas through borderline matches, and
grouping on D says whether two columns compute the same thing, not whether either holds.

**Where a direction comes from.** A features table carries the direction the arm's Stage 2 model
gave the feature before the reduction, because that is recorded for every arm. A drawn model shows
its own direction. The two can differ, and the component on few-feature models says so.

## The checks that make a figure trustworthy

A drawn model has to be the model the report quotes. Nothing is refitted: the model drawn is the one
the reduced cell saved, the one whose test predictions the reveal scored.

- **Our models** are loaded from the reduced cell's `fitted.pkl`, and their predictions for the
  validation rows must equal the scores the cell recorded when it was fitted, to within `1e-10`.
  Another pickle, other columns or other rows fail there.
- **DEFT's trees** are read on DEFT's side (`deft_cart.py tree`): the fitting rows are routed down
  the saved tree for each node's counts, every row's leaf must carry the tree's own prediction for
  that row, and the tree's validation predictions must equal the cell's.
- **IGSR's models** are loaded from the reduced cell's `model.pkl`, and their predictions for the
  validation rows, from terms IGSR's side evaluates, must equal the cell's, as ours must.
- **A PySR model** is one of ours: its `fitted.pkl` holds the equation as SeqMaestro's own tree, so
  it loads and predicts in Core without PySR. Its Shapley figure draws the values its fit kept and
  ranked its features by, and its equation is also a table file (`stage3a_pysr_equation`, built by
  `tables.equation` from the same export), with the names as text, constants to three
  significant figures, a product too long for a line continued on the next, a fraction too wide
  for a line written as a division on the line, and a factor too long for a line broken before a
  sign inside its brackets.

## Figures

Figures are drawn under `SeqMaestro_LocalUtils` from `figure_inputs.json` alone. The build under
Core decides every number, label and caption, so a figure cannot show a number the tables do not.
The palette is the validated light-mode reference from the project's data-visualisation guidance.
Three encodings are fixed across every figure:

- **Two categorical hues** for the contrast the design exists to make (features a language model
  wrote against features enumerated in advance), with a third for the reference line. A black-box
  reference model is drawn as an outline and gets no fourth hue.
- **A diverging blue-red pair** for polarity: the positive class of a binary target, or a higher
  value of a continuous one, in blue. Both poles are labelled wherever they appear, by the dataset's
  own class names (`dataset.json`, never defaulted), or by which way the target moves.
- **A single blue ramp** for magnitude, with light meaning near nothing.

Figures render for print in light mode, at aspect ratios that fit a portrait column. The figure
build measures each one and records whether it needs a rotated page.

## Changing the layer

**Expect a new dataset to expose something too narrow, and fix it here and not in the report.**
Examples are an explanation that describes only the dataset it was written against, a legend that
sits on the data when scores run high, or a table that fits the page only until a column is added.
Two rules make such a change safe:

- **A change belongs here only if it stays true on every dataset.** A component must not name a
  dataset, a biological entity, or a number one run happened to use. Where a number differs between
  runs, it is a parameter.
- **A change to a component, a table or a figure makes the reports already rendered inconsistent
  with the new one.** Rebuild them. A component or template change needs the assembly and render
  again, and a change to a table or a figure needs the inputs and figures rebuilt first.

A new model class or feature source needs an entry in `seqmaestro/report/naming.py`, which decides
how a reader sees it, in the report and in the spreadsheet of a run's features alike. An unknown
name raises, so it never reaches a reader as its internal name. A new interpretable model is also
drawn, so it needs a reader of its insides in `seqmaestro/report/internals.py` (`read`), an entry
in `DRAWN_AS` in `seqmaestro/report/figure_inputs.py`, and, where its fitted model is used outside
Core, a side in `seqmaestro/pipeline/sides.py`.

## What the build writes

Building a run's report takes four commands (`process/reporting.md`). The driver runs the first
two after the reveal. Everything goes under `<run>/report/`.

| file | written by | holds |
|---|---|---|
| `facts.json` | `build_report_inputs.py` (Core) | every number, keyed: the run, the kind of target and its metrics, the reveal, the classes or the target, Stage 1, Stage 2, the baseline, Stage 3A, Stage 3B, the two headline sets, the concepts, and `parameters`, the settings a report quotes with `{{param:NAME}}` |
| `tables/<name>.md` | the same | each table, rendered. A table of an earlier build that this one does not render is removed |
| `internals/<model>__<pool>.json` | the same | what is inside each drawn model, the best reduced arm of each interpretable class: its `kind` (`ebm_shapes`, `l1_weights`, `pysr_equation`, `tree` or `no_features`) and what that kind draws |
| `figure_inputs.json` | the same | for each figure, its drawer, caption, width and data |
| `figures/<name>.png`, `figures.json` | `build_report_figures.py` (LocalUtils) | each figure, and for each its path, caption, width, measured shape and whether it takes a rotated page |
| `REPORT.src.md` | the writer | the report, with placeholders |
| `REPORT.md` | `assemble_report.py` | the report with every placeholder filled |
| `REPORT.pdf` | `render_pdf.sh` | the rendered report |

The first two commands also leave their records with the run's other calls:
`invocations/report_inputs.json` and `invocations/report_figures.json`, and under the driver
`logs/report_inputs.log` and `logs/report_figures.log`.

**The tables.** An arm in a table's name is spelled `<model>__<pool>`.

| name | holds |
|---|---|
| `key` | each model and feature source of the run, with its abbreviation |
| `baseline` | the composition baseline: its score, its range over seeds, whether it fits the reduction's budget |
| `stage2_all_arms` | every arm: the held-out headline, its range, the same on shuffled labels, the gap, the features used |
| `stage2_headline` | the Stage 2 headline set, each against the baseline |
| `stage2_reproducibility` | every arm: reselection, recurring ideas, agreement across sources, and the judge's mean where the run was judged |
| `stage2_cross_arm_concepts` | ideas found across arms, on real and on shuffled labels |
| `stage3a_all_arms` | every reduced arm: the reduced and the full model's headline, its interval, the change |
| `features_<model>__<pool>` | one per reduced arm with a surviving column: each column's rank, description, direction and association on its own |
| `stage3a_shared_features` | the features several reduced arms kept, by name |
| `stage3a_pysr_equation` | a drawn symbolic model's equation, typeset, with what each of its features measures |
| `stage3b_summary` | every profiled arm: how many of its features pass each tier |
| `evidence_<model>__<pool>` | one per profiled arm: each feature its reduced model uses, with its stability, its held-out association and what removing it costs |
| `stage3b_tier2` | every profiled arm: the candidates tested and called |
| `stage3b_plug_in_fdr` | every profiled arm at each raw cut-off: the calls, the calls expected under the null, their ratio |
| `concepts`, `concepts_detail` | the ideas more than one arm reached, with their evidence, and what each computes |

**The figures**: `stage2_grid` and `stage3a_grid` (every arm's headline, by model and source),
`stage3a_cost` (what the reduction cost each arm), one figure per drawn model (`stage3a_ebm`,
`stage3a_l1`, `stage3a_igsr`, `stage3a_tree`, `stage3a_pysr` with `stage3a_pysr_front`),
`stage3b_evidence` and `stage3b_evidence_all`, `stage3b_thresholds` and `stage3b_concepts`.
`TEMPLATE.md` says where each table and figure goes.
