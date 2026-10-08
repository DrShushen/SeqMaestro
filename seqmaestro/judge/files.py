"""The files a judged run keeps under its `judge/` directory.

Imports nothing, because two environments use the names: Core writes the files
(`seqmaestro.pipeline.analyses.judge`), and the spreadsheet of a run's features, written under
`SeqMaestro_LocalUtils`, reads the scores.
"""

#: Each arm's mean rating, with the settings the panel was rated under and what the calls cost.
#: A report takes the judge's numbers from this file.
SUMMARY = "judge_summary.json"

#: One entry per feature on the panel: its arm, its label condition, its name, every rating it
#: got and the reason given with each.
SCORES = "judge_scores.json"

#: For each blinded item of the panel, the arm, label condition, seed and feature it was.
PANEL_KEY = "panel_key.json"
