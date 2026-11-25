"""The names of the files in a run's `report/` directory, kept here so that each is spelled in one
place, and the directory of the run-independent text a report includes.

It imports nothing beyond the standard library, because both environments import it: Core to write
the inputs, and LocalUtils to draw the figures.

    report/
        facts.json            every number, keyed, with the files it   } written by
                              was read from                            } build_report_inputs.py,
        tables/<name>.md      the same numbers rendered, in a reader's } under Core
                              names                                    }
        internals/<arm>.json  what is inside each drawn model: its     }
                              kind (ebm_shapes, l1_weights,            }
                              pysr_equation, tree or no_features) and  }
                              what that kind draws                     }
        figure_inputs.json    what each figure draws, with its caption }
                              and width                                }
        figures/<name>.png    the figures                              } written by
        figures.json          each figure's path, caption, width,      } build_report_figures.py,
                              measured shape, and whether it takes a   } under LocalUtils
                              rotated page                             }
        REPORT.src.md         the report as the agent wrote it, with placeholders
        REPORT.md             the report with every placeholder filled, by assemble_report.py
        REPORT.pdf            the rendered report, by render_pdf.sh
"""

from pathlib import Path

#: The run-independent explanations a report includes with `{{component:NAME}}`
#: (`seqmaestro.report.placeholders`). They are the same for every run, so they are kept with the
#: code that fills them in.
COMPONENTS_DIR = Path(__file__).resolve().parent / "components"

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
