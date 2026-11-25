# The analysis scripts

The scripts here work on a [run](../../docs/glossary.md#run) once [Stage 2](../../docs/glossary.md#stage) has fitted its [arms](../../docs/glossary.md#arm), or on a dataset and files that the caller chooses. This page says which question each script answers and when to use it. The pages under [`docs/developer/analyses/`](../../docs/developer/analyses/index.md) explain how each number is computed and what it can support, and [running experiments](../README.md) gives the rules that every script follows.

**A script that works on a run writes only files of its own**, under names that no stage uses, and changes nothing that a stage wrote, so running it again is safe.

**The Stage 2 analyses never open the [held-out split](../../docs/glossary.md#held-out-split).** They work from what each [cell](../../docs/glossary.md#cell) reported and from the values of its [features](../../docs/glossary.md#feature) on the validation rows, and they run before the [reveal](../../docs/glossary.md#reveal). A script that reports performance takes test scores from the cells only after the reveal, and it names the split its numbers come from ([the held-out split and the reveal](../../docs/developer/heldout.md)). `direct_effect_test.py` and `feature_group_dependence.py` work on the table they are given, so a table that holds the test rows opens the held-out split.

## The Stage 2 analyses

The [driver](../../docs/glossary.md#driver) runs these once every arm has finished, in this order, each with `--run-dir <run>`:

| Script | Question | Page |
|---|---|---|
| [`analyse_stability.py`](analyse_stability.py) | Does one arm report the same features again when it is fitted at another [seed](../../docs/glossary.md#seed)? | [Stability and co-discovery](../../docs/developer/analyses/stability_and_co_discovery.md) |
| [`analyse_co_discovery.py`](analyse_co_discovery.py) | Do different arms arrive at the same property? | [Stability and co-discovery](../../docs/developer/analyses/stability_and_co_discovery.md) |
| [`analyse_recurrence.py`](analyse_recurrence.py) | Does an arm on a [written source](../../docs/glossary.md#written-source) arrive at the same ideas again across seeds, and do different such arms arrive at the same idea? | [Recurrence](../../docs/developer/analyses/recurrence.md) |
| [`run_judge.py`](run_judge.py), where the run's [config](../../docs/glossary.md#config) sets `grid.judge: true` | How biologically significant would each feature be, if it held? | [The judge](../../docs/developer/analyses/judge.md) |

Each script's `--help` says what it writes. Every analysis takes the cells' [feature records](../../docs/glossary.md#feature-record) ([the feature record](../../docs/developer/feature_record.md)), and [direction](../../docs/developer/analyses/direction.md) explains the two directions that each record holds.

The [recurrence analysis](../../docs/glossary.md#recurrence-analysis) and the [judge](../../docs/glossary.md#judge) call a [language model](../../docs/glossary.md#language-model). Each keeps what it wrote while the features, the settings and the [task description](../../docs/glossary.md#task-description) it computed them from are unchanged, and `--force` makes it call the model again ([`inputs.py`](../../seqmaestro/pipeline/analyses/inputs.py)). Every analysis takes its settings from the run's recorded config, most of them from its `analyses` section ([run configs](../configs/README.md)), so a script started by hand behaves as it does under the driver.

## Stability, co-discovery and recurrence

All three measure whether something is found again, so they are easy to confuse. They differ in what they compare and in what counts as the same feature:

- **Stability** compares the seeds of one arm. Two features are the same when they have the same definition, or when their values correlate closely on the validation rows.
- **Co-discovery** compares different arms at one seed, with the same test of sameness, so that features computing one property match whatever each arm calls them.
- **Recurrence** compares ideas, across the seeds of one arm and across arms: a language model groups the written features that measure the same property, from their names and descriptions.

An arm whose features a language model writes can rename an idea and shift its window from one seed to the next, and two shifted windows do not correlate. Only recurrence can therefore show that such an arm returns to the same idea. Stability measures which features an arm selects, whereas recurrence measures what it proposes, and an arm can be consistent in one and not in the other ([recurrence](../../docs/developer/analyses/recurrence.md)).

Co-discovery and recurrence both call their groups [concepts](../../docs/glossary.md#concept). A concept of co-discovery is a group of features whose values agree, whereas a concept of recurrence is a group that a language model formed, so the two cannot be compared.

A figure from any of the three means little until it is compared with the same figure on the permuted cells, the fits on shuffled labels ([the permuted cells](../../docs/developer/analyses/index.md#the-permuted-cells)).

## Scripts run by hand

The driver does not start these, because each needs an argument that only the caller can choose, or runs in another environment. Each script's `--help` lists its arguments.

- **[`feature_group_dependence.py`](feature_group_dependence.py)**, before any arm is fitted, measures how much of each feature is the variable that the split groups on. A grouped split keeps the rows of a group together, but the features can still encode the group, and whether they do differs from one [pool](../../docs/glossary.md#pool) to another. A related question, what a design can detect at all, is measured when Stage 1 finalises the dataset ([Stage 1: the prepared dataset](../../docs/developer/dataset.md)), and [`design_power.py`](../prepare/design_power.py) answers it for a split that is still being planned.
- **[`direct_effect_test.py`](direct_effect_test.py)** tests whether one feature differs between the two classes, by how much, and whether a difference of that size could be chance, with a confounder matched and permuted within its levels. It is the script for a question about one feature of a report, such as "is that real?", and it computes the feature from the code that its cell recorded.
- **[`export_features_xlsx.py`](export_features_xlsx.py)**, under `SeqMaestro_LocalUtils`, writes every arm's reported features to a spreadsheet, for a collaborator who wants to sort and filter them. Its performance columns are on the held-out split once the run is revealed, and on validation before.
- **[`build_status_page.py`](build_status_page.py)** builds the run's [status page](../../docs/glossary.md#status-page) again after the driver has finished, for instance once the report is written.

[Direct measurement](../../docs/developer/analyses/direct_measurement.md) explains the first two.

## The report

After the reveal, four scripts build the report, in this order ([`process/reporting.md`](../../process/reporting.md) gives the commands, and [the report](../../docs/developer/report.md) says what the build decides):

1. [`build_report_inputs.py`](build_report_inputs.py), under `SeqMaestro_Core`, builds the facts, the tables and the inputs of the figures from the run.
2. [`build_report_figures.py`](build_report_figures.py), under `SeqMaestro_LocalUtils`, draws the figures from those inputs alone.
3. [`assemble_report.py`](assemble_report.py), under either environment, fills the placeholders of the `REPORT.src.md` that the report's writer wrote from [the report template](../../process/report-template.md), and writes `REPORT.md`.
4. [`render_pdf.sh`](render_pdf.sh), which needs `pandoc` and `xelatex` and no conda environment, renders `REPORT.pdf`.

The driver runs the first two.
