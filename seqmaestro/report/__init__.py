"""The report layer: turn a revealed run into the inputs a report is written from.

Nothing here computes a result. Every number comes from a stage that already produced it: a cell's
`metrics.json` after the reveal, a reduced arm's record, or a panel's tables. This layer reads those
numbers, renames them, tabulates them, and hands the figures what to draw.

| module | |
|---|---|
| `sources` | reads a revealed run back, from the paths its config lists (`Inventory`) |
| `naming` | the only path from an internal name to a reader's; an unknown name raises |
| `facts` | every number, keyed, with its source files; the headline-set rule; the run's parameters |
| `concepts` | the reduced models' features grouped by what they compute, across arms |
| `internals` | what is inside a drawn model, from the saved model itself |
| `tables`, `markdown` | every table a report may include, and how a number is printed |
| `figure_inputs` | what each figure draws, captions and widths included |
| `build` | writes all of the above under `<run>/report/`, with the file names `files` defines |
| `placeholders` | expands a report source into the finished markdown |
| `figures` | draws the figures from `figure_inputs.json`. **Needs matplotlib, so runs under
  `SeqMaestro_LocalUtils`**. Nothing else in `seqmaestro` imports matplotlib |
| `spreadsheet` | a run's reported features as a spreadsheet, apart from the report. Writing the
  file needs openpyxl, so it too runs under `SeqMaestro_LocalUtils` |

`docs/report/README.md` says what the layer decides and why. `process/reporting.md` is the writer's
instruction file.
"""
