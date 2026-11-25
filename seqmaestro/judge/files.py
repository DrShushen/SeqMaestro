"""The files a judged run keeps under its `judge/` directory.

This module imports nothing, because two environments use these names: Core writes the files
(`seqmaestro.pipeline.analyses.judge`), and the spreadsheet of a run's features, which is written
under `SeqMaestro_LocalUtils`, reads the scores.
"""

#: Each arm's mean rating, with the settings the panel was rated under and what the calls cost.
#: A report takes the judge's numbers from this file.
SUMMARY = "judge_summary.json"

#: One entry per feature on the panel: its arm, its label condition, its name, every rating it
#: got and the reason given with each.
SCORES = "judge_scores.json"

#: What each item of the panel stands for: the arm, label condition and seed it came from,
#: and the feature's name and importance.
PANEL_KEY = "panel_key.json"
