# The report

> This page tracks `seqmaestro/report/` with its components, and the commands that build a report: `build_report_inputs.py`, `build_report_figures.py`, `assemble_report.py` and `render_pdf.sh` in `experiments/analysis/`. Change it in the same commit as any of them.

A report is one document per [run](../glossary.md#run), written for a bioinformatician and a biologist reading the same page. It covers every [stage](../glossary.md#stage): how the data was prepared, which feature sources and model classes capture the signal, the few-feature models worth showing, and which of their features would survive a repeat of the study.

The [agent](../glossary.md#agent) writes the report's text, whereas the *report layer* ({py:mod}`seqmaestro.report`) prepares everything the text draws on: every number, the tables, the figures, and the explanations that are the same for every run. This page explains what the layer decides, and why it is built as it is.

**The report layer does not compute any result.** Every number comes from a stage that already produced it, and every number on the held-out split is one the [reveal](../glossary.md#reveal) scored. The layer gives those numbers a reader's names, tabulates them, draws them and assembles the document.

Two files and a folder shape a report besides the layer's code:

- [`process/reporting.md`](../../process/reporting.md): how a report is built, step by step, and what the agent does at each step.
- [`process/report-template.md`](../../process/report-template.md): the template, with the list of sections, what each must contain, the rules of style and the checks before the report is handed over.
- [`seqmaestro/report/components/`](../../seqmaestro/report/components): the *components*, explanations that do not depend on the run and that a report includes word for word, such as how an additive model works, what each tier of the reliability panel measures, and the glossary.

The first two guide the agent, and are not part of this site.

## The build

A report is built in five steps, which [`process/reporting.md`](../../process/reporting.md) gives as commands:

1. `build_report_inputs.py`, under `SeqMaestro_Core`, writes every number into `facts.json`, renders the tables, exports the contents of each model the report draws, and decides what each figure shows ({py:mod}`seqmaestro.pipeline.report`).
2. `build_report_figures.py`, under `SeqMaestro_LocalUtils`, draws the figures from what the first step decided.
3. The agent writes the report's source, `REPORT.src.md`, against the template.
4. `assemble_report.py` fills the source's placeholders, to give `REPORT.md`.
5. `render_pdf.sh` renders `REPORT.pdf`.

The [driver](../glossary.md#driver) runs the first two steps after the reveal. Each of them records its invocation with the run's other calls, and under the driver its log too ({py:mod}`seqmaestro.pipeline.layout`). The first takes from the run's config what the run should hold, so a missing cell stops the build instead of leaving a gap in the report, and it refuses a run whose reveal is incomplete ({py:mod}`seqmaestro.report.sources`).

{py:mod}`seqmaestro.report.files` lists the files each step writes under `<run>/report/`. The tables are listed in {py:mod}`seqmaestro.report.tables` and the figures in {py:mod}`seqmaestro.report.figure_inputs`, and the template says where each one goes in the report. A rebuild deletes any table or figure that the new build no longer makes, so that a stale one cannot be included.

## Four separations

**Numbers are kept apart from prose.** The build writes every number into `facts.json` and the rendered tables. A report's source is prose with placeholders, which the assembly fills in with a component, a table, a figure (with the caption and width the build recorded) or a setting the run chose ({py:mod}`seqmaestro.report.placeholders`). A report includes a table and says what it means, and never retypes a number, so the two cannot drift apart. An unresolved placeholder stops the assembly, so that it never reaches the reader as braces.

**Explanations are kept apart from findings.** A component says what a q-value is, and the report says what this run's q-values were. Improving an explanation improves every report, and an explanation is not rewritten for any one dataset. A number that a component needs and that differs between runs (how much resampling the panel did, how many features the reduction kept) is a parameter of the run, which the component quotes through a placeholder ({py:func}`~seqmaestro.report.facts.parameters`).

Where the kind of target changes what a component explains (the metric, the classes, the loss a refit is compared in), the component has one version per kind, `NAME.binary.md` and `NAME.continuous.md`, and the run's kind of target (`target_kind` in `facts.json`) chooses between them when the report is assembled. A component never switches between kinds inside its prose, so the reader of one kind of report meets nothing written for the other. A component that serves both kinds is one file, `NAME.md`, and says nothing that either kind would find wrong.

**The document is kept apart from the page it is printed on.** The report is plain markdown, which says nothing about page breaks or orientation, and a figure states only the share of the text width that the build recorded for it. `render_pdf.sh` renders the report in portrait on A4 with [pandoc](https://pandoc.org/), and a filter gives a rotated page to any figure that the figure build measured as too wide for a portrait column.

**Choosing what to show is kept apart from deciding what it means.** A fixed rule, applied in the same way to every run, decides which arms lead the report ([the headline set](#the-headline-set)), so a reader can see what the rule was. What the result means is the judgement of the agent who writes the report, marked as such in the document.

## The layer's decisions

### The headline set

The [headline set](../glossary.md#headline-set) holds the arms that the body of a report leads with ({py:func}`~seqmaestro.report.facts.headline_set`):

- The best arm of each interpretable model class that fits the kind of target. The sparse linear class is L1-penalised logistic regression for a binary target and the Lasso for a continuous one.
- The best interpretable arm whose pool has no [written source](../glossary.md#written-source), because the contrast between features enumerated in advance and features a language model wrote is the one the design exists to make.
- The strongest black-box arm, which shows how much signal the data holds.

The set is chosen twice: once on each arm's Stage 2 held-out score, averaged over the seeds, and once on the held-out score of its reduced model from Stage 3A. The two stages answer different questions, and an arm can lead in one but not in the other. The score is the headline metric of the kind of target, AUROC for a binary target and R² for a continuous one. Ties go to the arm whose name sorts first, so that the rule has one answer.

Whether a model class is interpretable is declared in {py:mod}`seqmaestro.arms` (`black_box`), which takes it from each Core model's `interpretable` attribute and states it for DEFT's tree and for IGSR. A tree can be followed node by node, so it counts as interpretable.

### The reference line

The [reference line](../glossary.md#reference-line) is the score of the composition [baseline](../glossary.md#baseline) ([the composition baseline](baseline.md)): its mean held-out score over the seeds, with its range. The facts also record whether the baseline's number of features is within the reduction's budget, which says whether a reduced model's comparison with the baseline is like for like.

### Words for the kind of target

`facts.json` names its keys after the metrics of the run's kind of target, so a key such as `reduced_auroc` of a binary run is `reduced_r2` of a continuous one ({py:mod}`seqmaestro.report.facts`). Its `directions` block says how a report prints each direction: as the class that a feature of a binary target points to, or as the way a continuous target moves.

### Concepts

[Concepts](../glossary.md#concept) group the reduced models' features across arms by what they compute, so that one idea found by several arms under different names counts once ({py:mod}`seqmaestro.report.concepts`). They use the matching and the clustering of the Stage 2 analyses ([comparing features by their values](analyses/index.md#comparing-features-by-their-values)), on the training and validation rows.

Every column that the reduction kept for a profiled arm is grouped, including a column its reduced model gives no weight. A Core arm's columns are computed by the reduced cell's saved featuriser, whereas DEFT's and IGSR's come from the reliability panel's evaluation over those rows, made on each method's side ({py:mod}`seqmaestro.pipeline.sides`).

The grouping has two limits, which any use of it should state. The chaining is transitive, so borderline matches can merge two ideas. And the grouping uses the rows the models were fitted and tuned on, so a concept shows that its members compute the same thing, and nothing about whether that thing holds.

### Directions in the tables

A table of a reduced arm's features shows the direction that the arm's Stage 2 model gave each feature, before the reduction, because that direction is recorded for every arm. The figure of a drawn model, by contrast, shows the direction in the reduced model itself. The two can differ, and the component on few-feature models says so.

## Checks on every drawn model

A drawn model has to be the model the report quotes, so the layer does not refit it. The model drawn is the one the reduced cell saved, whose test predictions the reveal scored, and it is drawn only after a check that it is that model: its predictions for the validation rows must equal the scores its cell recorded when it was fitted ({py:func}`~seqmaestro.report.internals.check_reproduces`). A DEFT tree is loaded on DEFT's side, which also checks that each fitting row's leaf holds the tree's prediction for that row, and an IGSR model's predictions come from terms that IGSR's side evaluates ({py:mod}`seqmaestro.report.internals`).

**A PySR model is a Core model.** Its `fitted.pkl` holds the equation as SeqMaestro's own expression tree, so it loads and predicts in Core without PySR ({py:mod}`seqmaestro.models.symbolic`). Its figure of [Shapley values](https://en.wikipedia.org/wiki/Shapley_value) draws the values its fit kept and ranked its features by, so the figure shows the numbers behind the reduction. Its equation is also a table, typeset from the same export with the features' names and with constants to three significant figures ({py:func}`~seqmaestro.report.tables.equation`), so a report never retypes it.

## Figures

The figures are drawn under `SeqMaestro_LocalUtils`, because matplotlib stays out of the environment that produces results ([environments](environments.md)). The drawing code takes its input from `figure_inputs.json` alone, whereas the build under Core decides every number, label and caption, so a figure cannot show a number the tables do not ({py:mod}`seqmaestro.report.figure_inputs`).

Every figure is rendered for print in light mode, at an aspect ratio that fits a portrait column, and follows one palette: two hues contrast written features with enumerated ones (with a third for the reference line), a diverging pair shows which way a feature points, and a single ramp shows magnitude. {py:mod}`seqmaestro.report.figures.style` holds the palette with the reasons for it. A black-box arm is drawn as an outline, without a hue of its own, since it is only a reference ({py:mod}`seqmaestro.report.figures.performance`).

## Changing the layer

**A new dataset can show that something in the layer is too narrow, and the fix belongs in the layer, where it serves every later report.** Examples are an explanation that describes only the dataset it was written for, a legend that covers the data when scores run high, or a table that fits the page only until a column is added. Two rules make such a change safe:

- **A change belongs here only if it stays true for every dataset.** A component must not name a dataset, a biological entity or a number that one run happened to use. Where a number differs between runs, it is a parameter.
- **A change to a component, a table or a figure makes the reports already rendered inconsistent with new ones**, so rebuild them. A change to a component or to the template needs the assembly and the render again, and a change to a table or a figure needs the inputs and the figures rebuilt first.

**A new model class or feature source** needs an entry in {py:mod}`seqmaestro.report.naming`, which decides how a reader sees it, in the report and in the spreadsheet of a run's features. An unknown name raises an error, so that it never reaches a reader as its internal name.

**A new interpretable model** joins the headline set, so its reduced model is drawn, and it needs:

- A function that exports its contents, called from {py:func}`~seqmaestro.report.internals.read`.
- An entry in `DRAWN_AS` in [`figure_inputs.py`](../../seqmaestro/report/figure_inputs.py), which names its figure and the drawer that draws it.
- A side in {py:mod}`seqmaestro.pipeline.sides`, where its fitted model is used outside Core.
- A component that explains how the model works, and a line for the model in the template's section on the reduced models.
