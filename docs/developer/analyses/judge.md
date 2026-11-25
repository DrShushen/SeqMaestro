# The judge

> This page tracks `seqmaestro/pipeline/analyses/judge.py`, which `run_judge.py` calls, and `seqmaestro/judge/`. Change it in the same commit as any of them, and above all with the prompt's text in `seqmaestro/judge/rubric.py`, since a change to that text changes what past scores mean.

The [judge](../../glossary.md#judge) is a language model that rates how meaningful each of a run's most important features would be if it held. It is shown one feature at a time, without being told which [arm](../../glossary.md#arm) produced it, and asked:

> How biologically significant or interesting would this feature be, if it were true?

The other Stage 2 analyses measure whether a feature comes back, at another seed or in another arm, whereas the judge rates whether anyone should care about it. It runs only where the run's [config](../../glossary.md#config) sets `grid.judge` ([running the judge](#running-the-judge)).

## The question

**"If it were true" is the crucial part.** It separates significance from correctness, so that the judge is not asked to decide truth from a description it cannot verify, which it would do badly while sounding confident.

The same words sharpen the negative control. A feature derived from noise can be interesting if true, and the question allows that. Permuted features that score as highly as real ones are therefore informative: they show that the score follows how a feature is written, whatever produced it.

**One question serves better than a rubric of parts.** Parts such as mechanistic plausibility, specificity, standing relative to known biology and testability mostly vary together. Averaging them would give a number that cannot be interpreted, and reporting them separately would quadruple the output without changing any decision.

**The scale is anchored.** Each score from 1 to 5 has a sentence that says what it means (`SCALE` in [`rubric.py`](../../../seqmaestro/judge/rubric.py)), because an unanchored scale drifts between calls and cannot be compared across arms. The system prompt (`SYSTEM_PROMPT`, in the same file) also tells the judge to be discriminating, since most candidates are ordinary and a scale on which everything scores 4 measures nothing.

## The panel

The features the judge rates form one *panel*, which {py:mod}`seqmaestro.judge.panel` assembles in three steps. Each step is easy to get subtly wrong, so it is kept in the library and out of the script:

1. **Selection.** The panel holds the [top N](index.md#the-top-n) features of each real-label [cell](../../glossary.md#cell), so an arm puts up to N features on the panel at each seed, and a feature it uses at every seed is rated once for each. Judging every feature is unaffordable, and an arm does not claim to have discovered the tail of a vocabulary of thousands.
2. **Hiding the source.** Each entry gets an identifier that reveals nothing, such as `F0007`. The arm, the seed and the label condition are kept beside the panel, never inside it, so a prompt cannot mention them. This holds by construction, without relying on anyone to remember it, and `tests/test_judge.py` checks that no arm's name reaches a prompt.
3. **Ordering.** The entries are shuffled with a seed, so that an arm's features are not adjacent and a judge that drifts cannot systematically favour whichever arm came first. The identifiers are assigned after the shuffle, since identifiers assigned before it would give each arm a block of consecutive identifiers, which would reveal the arm.

**Only real-label features are on the panel by default.** A config whose `analyses.judge.labels` includes `permuted` adds the permuted cells' features. They enter as ordinary entries, which the judge cannot tell apart from the rest, and are separated again only when the scores are summarised. [Limits](#limits) says what such a control can show, and why it is not judged by default.

The judge's files under `<run>/judge/` are named, each with what it holds, in [`seqmaestro/judge/files.py`](../../../seqmaestro/judge/files.py). The panel's key says what each identifier stands for, and the scores name each item's arm and label condition, since the summary is taken by arm. Neither file hides anything, because only the judge has to be kept from knowing where a feature came from.

## The prompt

{py:func}`~seqmaestro.judge.rubric.render` builds each prompt. It holds the task description, the question, the feature's name and description, its stated rationale where the record has one, and the scale with the form of the reply. Two settings of the run each add a line:

- `grid.judge_direction` adds which way the feature runs in the data ([the direction line](#the-direction-line)).
- `analyses.judge.show_code` adds the feature's code. The shared config leaves it off, since code may help the judge with an ambiguous description, or bias it towards whatever looks sophisticated.

### Blinding

**The feature's origin is hidden**: the judge is not told which arm produced the feature or how it scored.

**In a [blinded](../../glossary.md#blinding) run, by contrast, the judge is told more than the other language models.** The [task description](../../glossary.md#task-description)'s `judge_only` field holds a fact that the judge is told and that every language model that writes features is not. The [proposer](../../glossary.md#proposer) and the methods that write features are what the experiment tests, so a fact they could retrieve from what they already know (the name of the protein whose loss defines the classes, say) would make the result uninterpretable if they were given it. The judge is the instrument that measures them, and it cannot say how significant a feature would be without knowing what the phenotype is.

**The plain rendering of a task description is the blinded one.** A caller that uses `render` gets what the models that write features get, whereas the judge's prompt uses `render_for_judge` ({py:class}`~seqmaestro.llm.context.TaskContext`). A caller who forgets which rendering it wanted therefore discloses too little and reveals nothing.

### The direction line

The direction line is a setting of the run, `grid.judge_direction`, which the shared config turns on. The judge takes it from the run's recorded config, however it is started, so every rating of a run is made under one setting.

The line changes the question. For a count of upstream start codons, which is plausible either way round, the question can barely be answered. A judge told which way the feature runs, by contrast, scores a specific claim, and can mark one down that contradicts what is known. That is a better instrument and a different one, so scores from a run with the line cannot be compared with scores from a run without it, which is why the summary records the setting.

**The judge is shown the sign of the effect, never its size.** An effect size would let the judge tell real features from the permuted cells' features, where those are judged as a control, whereas a sign is a coin flip for a control's feature, so it tells the two apart no better than chance. For the same reason, the direction shown is the feature's marginal direction in the data, from its `marginal_effect` ([direction](direction.md)). That is a statement about the data, which is what the question concerns, and it is always one of two words, whereas the model's direction allows a third.

The summary also records how many features had a direction to show (`features_with_direction`), because asking for directions and getting none would otherwise pass silently. A feature with no `marginal_effect` is sent without a direction line, so the setting could be on while no prompt had the line.

## Running the judge

A config says whether the judge runs, in `grid.judge`, which the shared config ([`_shared.yaml`](../../../experiments/configs/_shared.yaml)) turns off. It is off there for two reasons. Each rating is a call to the language model, and the judge rates the top N features of each real-label cell several times each (`analyses.top_n` and `analyses.judge.repeats`), so its calls grow with the arms and the seeds. And the stages after Stage 2 do not depend on its scores. A dataset's config sets `judge: true` to run it. A run without the judge has no judge stage, and its report has no column for the judge.

Every setting of the judge comes from the run's recorded config, and the script has no flag for any of them, so a judge started by hand rates as the driver's does. A real run needs the language model's settings in the environment (`set -a && source .env && set +a`), whereas `--dry-run` builds the panel, prints the first prompt and makes no call. Every call is made under the config's `language_model` settings (the reasoning effort, the retries and the wait), and the summary records which model was asked and how (`language_model`).

The judge keeps its files when it is started again while what they were computed from is unchanged: the features, the settings, the task description, the model and its reasoning effort ({py:mod}`seqmaestro.pipeline.analyses.inputs`). `--force` rates the features again.

## Scoring

**Every feature is rated `analyses.judge.repeats` times, in independent calls**, so the spread across the repeats measures how reproducible the judgement was.

**A refusal gives no score.** A call that fails, times out or returns something that cannot be parsed counts as unanswered (`n_unanswered` in the summary). Treating a failure as a middling score would quietly pull every arm towards the mean, and would flatter most the arms whose features are hardest to parse. The stage fails when less of the panel was answered than `analyses.judge.min_answered` requires, because a mostly unanswered panel is a failed stage, with no result to report.

{py:func}`~seqmaestro.judge.engine.parse_reply` tolerates how a reply is wrapped, in prose or in a fenced block, but never invents a score, and its docstring gives the reasons.

**The summary** gives each arm's mean under each label condition, with the spread across repeats and the number of unanswered calls ({py:func}`~seqmaestro.judge.scoring.summarise`). Where the permuted cells' features were judged, their mean over every item is the level this instrument returns when there is nothing to find (the *noise floor*), and each arm's distance above it is `control_gap`. As an illustration of the arithmetic, an arm that scores 3.6 against a floor of 3.5 has shown nothing, whereas the same 3.6 against a floor of 2.4 is a result. A run judged on real-label features alone has no floor and no `control_gap`, and its means compare the arms with each other.

## Limits

**The judge scores writing as much as biology.** A feature with a fluent description and a plausible rationale scores well whether or not it came from real labels. Judging the permuted cells' features measures this for an arm that selects from a fixed vocabulary, because its permuted features are different features: if they score as highly as the real ones, the judge is scoring how a feature is written.

**The permuted cells' features cannot serve as a control for an arm on a [written source](../../glossary.md#written-source), so they are not judged by default.** The proposer takes its features from the task description and its own knowledge, and the labelled rows are a small part of what it is shown. On shuffled labels it proposes nearly the same features, so the judge would score the same panel twice, and the number would look like a control without being one. The recurrence analysis shows this from the other side, by counting each idea's features under both label conditions for every arm on a written source ([recurrence](recurrence.md)). A single arm's mean therefore describes its features, and says nothing about the labels. The control the judge needs is a shuffled task description, which SeqMaestro does not build.

**A high score is no evidence that the feature is true**, since the question sets truth aside. Pair a high score with [direct measurement](direct_measurement.md) and with recurrence before treating a feature as a finding.

**The judge is told the task description, which states facts and says nothing of known confounds.** Telling the judge that GC content is confounded in a dataset would have it mark GC features down, which is a judgement about whether a feature is real, the very thing that "if it were true" sets aside. Such warnings belong in the report.

**A change to the prompt invalidates comparisons.** The text of the rubric, the wording of the scale, `analyses.judge.show_code`, `grid.judge_direction`, the model and its reasoning effort all change what a score means. A change to any of them therefore means rating again every arm being compared, including those rated before the change.
