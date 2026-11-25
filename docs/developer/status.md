# The status page

> This page tracks {py:mod}`seqmaestro.status` and the driver's use of it ({py:mod}`seqmaestro.pipeline.driver`). Change it in the same commit as either.

Each [run](../glossary.md#run) has a [status page](../glossary.md#status-page), `status.html` at the run's root, which shows where the run stands and what each [stage](../glossary.md#stage) has found so far. It is for the person who started the run and for the [agent](../glossary.md#agent) following it. At a glance, they see which stage is running, which [arms](../glossary.md#arm) and cells are done, whether a [checkpoint](../glossary.md#checkpoint) has failed, and where the files behind each number are. [Monitoring a run](../user/running.md#monitoring-a-run) describes the page from the user's side.

```{figure} ../user/status-page.png
:alt: The status page of a run during Stage 3B
:width: 100%

This is the status page of a quick run of the example dataset, during Stage 3B. The band above the stages shows the run as a whole. Each column shows one stage, with its details collapsed and its checkpoint at the foot.
```

The page has a band above the stages for the run as a whole, one column for each of Stages 1, 2, 3A and 3B, and a band below for the report. {py:mod}`seqmaestro.status.gather` has one function per part, whose docstring says what the part shows.

## Design rules

- **The page is built from the run's own records**: the files the run writes, and the scheduler's snapshot of the run's jobs (`logs/jobs.json`). It therefore shows no number that the run does not record, and building it again gives the same page.
- **Its checkpoints restate the checks of the agent's instructions** ([`process/`](../../process/README.md)), computed from the run's files, so a check that changes in `process/` changes on the page in the same commit.
- **A control cell is marked where its validation interval lies above chance**, because that is how a leak shows. A model fitted to shuffled labels can fall below chance by picking a direction along one feature, so a control below chance is left unmarked.
- **The page names arms, models and feature sources as the report does** ({py:mod}`seqmaestro.report.naming`), so that the page and the report use one vocabulary.
- **The report's figures appear in each stage's column once they are drawn**, which is after the reveal. Until then, the column shows the numbers the figures are drawn from.

## Keeping the page current

The page is one static file. It needs no server, opens from the run's folder in any browser, and still works when the folder is copied elsewhere, because its stylesheet is inlined and its links are relative.

The driver keeps the file current: it writes the page as the run starts and stops, as processes start and end, and at intervals while a long process runs ({py:mod}`seqmaestro.status` gives the intervals). While the driver runs, an open page reloads itself. A failure to build the page is written once in the driver's log and stops nothing, since the page only reports. After the run, [`build_status_page.py`](../../experiments/analysis/build_status_page.py) `--run-dir <run>` builds the page again, as the agent does once the report is written.

## Changing the page

{py:mod}`seqmaestro.status.gather` collects what the page shows, as plain values. The template [`page.html`](../../seqmaestro/status/templates/page.html) lays them out, in the colours of [`page.css`](../../seqmaestro/status/templates/page.css), which are those of the documentation's figures. A new item takes three steps:

1. Add it to the function of its part in {py:mod}`seqmaestro.status.gather`. Take each file's name from the module that names the file, so that renaming a file cannot leave the page looking for the old name.
2. Show it in that part's block of the template.
3. Test it in [`tests/test_status.py`](../../tests/test_status.py), on a run directory built for the test.
