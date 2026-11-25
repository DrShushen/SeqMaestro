"""The report layer: turn a revealed run into the inputs a report is written from.

Nothing here computes a result. Every number comes from a stage that already produced it: a cell's
`metrics.json` after the reveal, a reduced arm's record, or a panel's tables. This layer reads those
numbers, renames them, tabulates them, and hands the figures what to draw.

| module | |
|---|---|
| `sources` | reads a revealed run back, from the paths its config lists (`Inventory`) |
| `naming` | the only path from an internal name to a reader's, which raises on an unknown name |
| `facts` | every number, keyed, with its source files, as well as the headline-set rule and the run's parameters |
| `concepts` | the reduced models' features grouped across arms by what they compute |
| `internals` | what is inside a drawn model, taken from the saved model itself |
| `tables`, `markdown` | every table a report may include, and how a number is printed |
| `figure_inputs` | what each figure draws, captions and widths included |
| `build` | writes all of the above under `<run>/report/`, with the file names `files` defines |
| `placeholders` | expands a report source into the finished markdown |
| `figures` | draws the figures from `figure_inputs.json`, under `SeqMaestro_LocalUtils` |
| `spreadsheet` | a run's reported features as a spreadsheet, apart from the report |

**`figures` needs matplotlib, so it runs under `SeqMaestro_LocalUtils`**, and nothing else in
`seqmaestro` imports matplotlib. Writing the spreadsheet needs openpyxl, so `spreadsheet` runs
under `SeqMaestro_LocalUtils` too.

`docs/developer/report.md` says what the layer decides and why. `process/reporting.md` is the
instruction file of the agent that writes a report.
"""
