# Troubleshooting

## The installer stops with an error

Its last lines say which step failed. Fix what they name, then run `./install.sh` again, which continues from what is already built.

## The check reports a problem

Each line of `./install.sh --check` marked `PROBLEM` ends with the command that fixes it. To rebuild one environment from scratch, remove it and build it again, for example:

```bash
conda env remove -n SeqMaestro_Method_pysr
./install.sh --only pysr
```

## XGBoost cannot be imported on macOS

If the check says that `SeqMaestro_Core` cannot import XGBoost, the OpenMP runtime is missing. Install it with `brew install libomp`, then run `./install.sh --check` again.

## The call to the language model fails

`./install.sh --check-llm` prints the provider's own error message. The usual causes are:

- **With Azure:** an endpoint without `https://` or without its final `/`, the model's name given where the deployment's name is needed (the two can differ), or an API version the deployment does not support.
- **With the OpenAI and Claude APIs:** a model's name that the key cannot use.
- **With Azure OpenAI, the OpenAI API or a current Claude model:** a model that refuses the reasoning effort, which the error message says. Set an effort the model accepts as `language_model.reasoning_effort` in `experiments/configs/_shared.yaml` ([call settings](language-model.md#call-settings)).

## Codex refuses a write, or PySR fails under Codex

Codex runs each command in a sandbox that may write only in the repository's folder, to which `./agent.sh codex` adds the two folders that PySR's Julia writes to. Started any other way, Codex cannot run PySR, which fails with `Read-only file system`, so start Codex with `./agent.sh codex`. A command that has to write anywhere else runs outside the sandbox once you approve it.

## OpenHands says the Responses API needs a newer API version

The message *Azure OpenAI Responses API is enabled only for api-version 2025-03-01-preview and later* means that the agent's model on Azure is called with an older API version. In Agent Canvas, set **API Version** to `2025-04-01-preview` under **Settings > LLM**, tab **All**.

## Claude Science's commands fail with "Read-only file system"

Claude Science was started with its sandbox, which gives code read-only access to the repository's folder. Stop it with `claude-science stop`, and start it again with `./agent.sh claude-science`, which turns the sandbox off ([Claude Science](agents.md#claude-science)).

## A run stopped before it finished

Ask the agent to continue it ([monitoring a run](running.md#monitoring-a-run)), or give the launch command again with `--run-dir` and `--stop-before-reveal`, as [running by hand](by-hand.md#running-stages-2-to-3b) shows. Every stage resumes from what is on disk.

## There is no PDF of the report

The PDF needs pandoc, XeLaTeX and the DejaVu fonts, and [PDF tools](installation.md#pdf-tools) says how to install them. Once they are installed, render the report of a finished run:

```bash
bash experiments/analysis/render_pdf.sh experiments/runs/<run>/report/REPORT.md
```
