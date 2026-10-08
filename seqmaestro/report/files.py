"""The names of the files in a run's `report/` directory, kept here so each is spelled in one place.

Imports nothing, because both environments read it: Core writes the inputs, LocalUtils draws the
figures.

    report/
        facts.json            every number, keyed, with the files it was read from
        tables/<name>.md      the same numbers rendered, in a reader's names
        internals/<arm>.json  what is inside each drawn model
        figure_inputs.json    what each figure draws
        figures/<name>.png    the figures, drawn under SeqMaestro_LocalUtils
        figures.json          each figure's path, caption, width and measured aspect
        REPORT.src.md         the report as written, with placeholders
        REPORT.md, REPORT.pdf the report expanded, and rendered
"""

FACTS_FILE = "facts.json"
TABLES_DIR = "tables"
INTERNALS_DIR = "internals"
FIGURE_INPUTS_FILE = "figure_inputs.json"
FIGURES_DIR = "figures"
FIGURES_FILE = "figures.json"
SOURCE_FILE = "REPORT.src.md"
REPORT_FILE = "REPORT.md"

#: The name of the call that draws a report's figures (`seqmaestro.pipeline.layout.call_name`).
#: It is stated here because the script runs under `SeqMaestro_LocalUtils`, where the driver and
#: the stages of `seqmaestro.pipeline` cannot be imported: they read a run's config, which needs
#: `pyyaml`.
FIGURES_CALL = "report_figures"
