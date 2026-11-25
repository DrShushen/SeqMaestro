# Running from the command line

The command `seqmaestro` takes a table of sequences through a [run](../glossary.md#run) without an agent. It prepares your data with fixed defaults, and asks about your data only where it needs to:

- Which column holds the sequences, and which holds the measurement.
- Which value marks the positive class, where there are two classes.
- A few sentences that describe the data.

It then starts the run, follows it until it ends, and has the report written. The language model that `.env` names drafts the [task description](../glossary.md#task-description) from your sentences, and writes the report.

[Running with an agent](running.md) is another way through a run. There, the agent decides each step with you, can prepare data in more forms, and checks the results with you before the held-out data is opened ([limitations](#limitations)).

Like the rest of SeqMaestro, the command runs on Linux and macOS, and on Windows inside WSL.

## Starting a run

The installer puts the command in the `SeqMaestro_Core` environment. Activate that environment, and give the command your table:

```bash
conda activate SeqMaestro_Core
seqmaestro run my_sequences.csv
```

If conda is not set up in your shell, run the same command with the environment's Python, from the repository's folder:

```bash
eval "$(./install.sh --interpreters)"
$CORE -m seqmaestro.cli run my_sequences.csv
```

The command asks one question at a time, with a list to choose from where the possible answers are known, and shows what it found after each answer. It writes its files into the repository's folder, wherever you start it from.

## Input table

The command takes one CSV or TSV file (`.csv`, `.tsv` or `.tab`, optionally gzipped), with a row for each sequence and these columns:

- A column of DNA sequences (`A`, `C`, `G`, `T`) or RNA sequences (`A`, `C`, `G`, `U`), in upper or lower case, with `N` for an unknown base.
- A column of what was measured for each sequence: either two values for two classes, such as `bound` and `unbound` or 1 and 0, or a number.
- Optionally, columns that name the rows that belong together, such as a gene or a region. The split keeps such rows on the same side.

Any other column is left out of the dataset, although the table itself is copied unchanged into the dataset's folder.

## The steps

The command goes through eight steps, each under its own heading in the terminal:

| Step | Questions | Output |
|---|---|---|
| 1. The table | Where the table is, and the dataset's name, which names its folder and its run | The table's first rows |
| 2. The columns | The column of sequences, the column of what was measured, the value of the positive class (for two classes), and the columns that name rows belonging together | Whether the target has two classes or is a number |
| 3. The rows and the split | If the sequences differ in length, the end of each sequence that positional features count from | The rows kept and dropped, and the rows of each split |
| 4. What the models are told | A few sentences about the data, from which the language model drafts the task description and the meaning of each class. You can use the draft, edit it in your editor, or describe the data again | The draft |
| 5. The dataset | None | The files it writes, and what the dataset's checks found |
| 6. The run | Which run to make, `quick` or `full`, and whether to start it | The estimate of each arm's time |
| 7. Monitoring the run | None | Each stage's progress and what is running, until the run ends |
| 8. The report | None | Its progress while the language model writes the report and the report is put together and rendered |

Without a language model configured in `.env`, step 4 writes the description from your sentences and from the table, and you edit it. The task description is what every language model of the run is told about your data. The report also uses each class's meaning to say which class a feature points to, so check both with care.

## Defaults

The agent would put each decision below to you, whereas the command line takes it as the table shows, and records it in the dataset's [declaration](../glossary.md#declaration) and README:

| Decision | Default |
|---|---|
| The rows | A row with no value in the target column is left out. A row whose sequence is missing, or has a symbol other than `N` outside its alphabet, is dropped, and the drop is recorded as a filter step |
| The sequences | Used as delivered, apart from conversion to upper case. The command does not take a window around a landmark or check the orientation |
| Sequences of different lengths | Positional features cover as many bases as the shortest sequence has, counted from the end you choose, so that no sequence is padded or dropped. Each k-mer count is divided by the number of positions in its sequence, so that it does not grow with the length |
| The split | Ten folds: seven for training, one for validation and two held out. The rows of one group, and rows with identical sequences, stay on the same side. The folds are balanced on the class, or on five equal-sized bins of the number |
| What the models are told | Everything in the task description: the run is not [blinded](../glossary.md#blinding) and has no [guidance](../glossary.md#guidance) on the form of the features |
| The feature sources | Positions and k-mers, with the [written sources](../glossary.md#written-source) as well in the full run. The motif sources and covariates are not used |

## Quick and full runs

| Run | Arms | Time | Language model |
|---|---|---|---|
| `quick` (the default) | The arms on positions and k-mers, without PySR, at two seeds, with a small reliability panel | Minutes | Not called by the run |
| `full` | Every arm of the shared config, at five seeds, with the full reliability panel | Hours | Called by the arms whose features a language model writes, and by the analysis that groups those features |

Before the run starts, the command shows the estimate of each arm's time on your machine and, where an arm calls the language model, its calls and cost. Both runs go through the [reveal](../glossary.md#reveal): the held-out rows are opened once every stage before the reveal has finished without a failure, and the report's tables and figures are then built.

## Monitoring the run

The command shows the run's progress until the run ends:

```{figure} command-line.svg
:alt: The command's view of a quick run of the example dataset during Stage 2
:width: 100%

This is the command's view of a quick run of the example dataset, during Stage 2.
```

Press Ctrl+C to leave the view. The run goes on in the background after the terminal closes. To see where a run stands later:

- `seqmaestro status` shows the run that changed last.
- `seqmaestro status <run>` shows another run, named by its folder or by the folder's name under `experiments/runs/`.

The run's [status page](../glossary.md#status-page), `status.html` in its folder, shows the same in more detail ([monitoring a run](running.md#monitoring-a-run)). When the run has finished, its folder holds what a run made with the agent holds: the fitted models, the tables, and the report's tables and figures in `report/`.

## The report

Once the run has finished without a failure, the language model writes the report in one call, following the template that an agent follows ([`process/report-template.md`](../../process/report-template.md)). The report has the template's sections, with the run's own tables, figures and explanations put in where the template places them, and the model's interpretation in the marked notes.

The report is `report/REPORT.md` in the run's folder, with `report/REPORT.pdf` beside it if the [PDF tools](installation.md#pdf-tools) are installed. The report's table of agents, which names who carried out each stage, names the command line and the model that wrote the report.

To have the report of a finished run written, or written again:

```bash
seqmaestro report <run>          # the run's folder, or its name under experiments/runs/
seqmaestro report <run> --again  # replaces the report written before
```

Without a language model configured, the command stops at the report's tables and figures. An agent can then write the report: start one and ask it to report on the run. An agent opens the run's files and checks its draft against them as it writes, which the command line's single call to the model cannot do.

## Command options

Every question has an option, and a question answered by an option is not asked. `--yes` takes the default answer of every other question that has one, and starts the run without asking. A question about your data has no default, so under `--yes` the command stops at the first such question left unanswered, and names its option. For example:

```bash
seqmaestro run my_sequences.csv --name my_sites --sequence sequence --target bound --positive 1 \
    --group-by region --describe "Each row is a candidate binding site, labelled 1 where the factor bound." \
    --preset quick --yes
```

| Option | Question | Default under `--yes` |
|---|---|---|
| `TABLE` | The table | None |
| `--name` | The dataset's name | The table's file name |
| `--sequence` | The column of sequences | None |
| `--target` | The column of what was measured | None |
| `--positive` | The value of the positive class, for two classes | `1` where the values are 0 and 1, and `true` where they are false and true |
| `--group-by` | A column that names rows belonging together (repeat it for more) | None |
| `--anchor` | The end positional features count from, `start` or `end` | `start` |
| `--describe` | A few sentences about the data | None, and the description is written from the table |
| `--preset` | The run, `quick` or `full` | `quick` |

`seqmaestro run --help` lists them too.

## Output files

The command writes the same files as the agent writes in Stage 1, in the same places, so that an agent or the driver can take any of them further:

| File | Contents |
|---|---|
| `data/<name>/` | The dataset's README, the table as given (`raw/`), and the splits with their `dataset.json` |
| `experiments/prepare/declarations/<name>__splits.yaml` | What was decided about the dataset |
| `experiments/datasets/<name>.json` | The task description the language models are told |
| `experiments/configs/<name>.yaml` | The run's config |
| `experiments/runs/<date>_<name>_full/` | The run |

`seqmaestro run` never replaces a file, so each dataset name can be used only once. To change a setting the command does not ask about, edit the config and start the driver yourself ([running by hand](by-hand.md)).

## Limitations

Because the command line takes a single table as it is, prepare your data with the agent when it needs more, for example:

- Several files to join.
- A window taken around a landmark, such as a splice site.
- Measurements beside the sequence, to use as covariates.
- A filter or a transformation of the target.
- A fact withheld from the models that write features (blinding).
- Guidance on the form of those features.
- The motif sources.

The agent also checks the results of the run with you before the held-out data is opened.
