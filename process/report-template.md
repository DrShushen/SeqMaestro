# Consolidated report template

**You are writing `REPORT.src.md` for one run, from the four stages of the pipeline.** Keep the
section list, the order and the headings below. What the commentary says is up to you, whereas
what each section must contain is fixed. `process/reporting.md` says how the inputs are built and
how the source is expanded and rendered.

---

## Your inputs

Everything you need is in the run's `report/` directory before you start, plus the dataset's own
records, which you read yourself.

| input | what it is | how to use it |
|---|---|---|
| `report/facts.json` | every number from all four stages, keyed and structured, each arm's entry naming the files it came from | the source of any number you quote in prose. Performance is on the held-out split unless a key says `val_`. Its `parameters` block holds the settings a run chose, which differ between runs, so refer to those in prose with `{{param:NAME}}` rather than writing the number |
| `report/tables/*.md` | the same numbers already rendered as markdown tables, in reader-facing names | **include verbatim.** Never retype a number out of one |
| `report/figures.json` | the figures, with the caption and the width the build recorded for each | include one with `{{figure:NAME}}`. Never write an image path or a caption by hand |
| `seqmaestro/report/components/*.md` | run-independent explanations | **include verbatim** where the section list says so. Each supplies its own heading, so do not write one above it. Three have none: the glossary, and `33_model_reference` and `61_stage2_judge`, each of which continues the component included before it. A component that explains something the kind of target changes has a twin per kind (`NAME.binary.md`, `NAME.continuous.md`). Include it by its name alone, `{{component:NAME}}`, and the run's kind of target chooses the twin |
| the dataset's `README.md` | the preparation decisions, in the builder's words | read it, and rewrite Stage 1 from it |
| the dataset's `dataset.json` | the target (its classes and what each means, or a continuous target's name and meaning), the split, every preparation step with the rows it removed, the checks, the detection floor | `facts.json`'s `stage1` block holds it, and `facts.json`'s `classes` or `target` the target. The class names, or the target's name, come from here and nowhere else |
| the run's `config.yaml` | what the run was configured with. Its `dataset.task` names the description the language models were told, `experiments/datasets/<task>.json`. Its `description` is a note on why the variant exists, which no model was told | read the description file for what the models were told, and on a blinded run its `judge_only` for what was withheld from them |

**Do not compute a new number.** If a number you want is missing, say so or leave it out. A report
whose arithmetic is done in two places will eventually disagree with itself.

**A placeholder is substituted verbatim**, so do not start a sentence with one: it would need a
capital and would not get one.

---

## Who reads this

A bioinformatician and a biologist who does not model read the same document, so every
machine-learning term is defined where it first appears, and every biological claim stays
checkable. Neither reader should have to ask us what something means.

**Neither reader has seen this project before**, so use no internal vocabulary, code paths or
directory names, and do not mention our tooling, our costs, our environments or anything we would
fix.

---

## The angle

The pipeline delivers two different things, and both are results.

1. **A model** you could apply to a new sequence, small enough to interpret end to end.
2. **A set of ideas** that held up across independent arms and on sequences nothing was fitted
   on, which is what a follow-up experiment is designed around.

These are different goals, so one feature need not serve both. Write the report that way. A
feature can be in the best model without being the most dependable one, and a dependable feature
can be absent from the best model because a near-duplicate was already there.

**Do not write the report as a model followed by a list of reasons to distrust it.** Where the
reliability panel is silent about a feature, say in a line what it is silent about and why, and
move on. Where a study lacked the power to resolve something, state that once, as a fact about the
study's size. Do not hedge every claim, because hedging buries the findings that did survive.

---

## Fact and interpretation are kept apart

Everything outside a marked block is a number or a statement of what was done, traceable to a table
or a figure you included.

Interpretation, and anything that would change if a different reader looked at the same information,
goes inside a block marked:

> **AI Agent Analysis Note**
>
> ...

Every stage section ends with one such block, and only one, even when there is little to say. The
closing "What this suggests looking at next" section is a single such block from start to finish.

---

## Style

Follow all of these.

- Every sentence carries meaning. Leave out filler.
- No "not X but Y", and no telling the reader what something is not.
- **No em dashes or en dashes.** Use commas, colons, parentheses or a full stop.
- **No "X's own Y" phrasing.** Say what the thing is instead.
- **No "is exactly Z" phrasing.** Say "is Z".
- No chains of present-tense clauses joined by semicolons. Use connectives and stated logic.
- No "and it is worth explaining why", no "and here is the thing". Just say the thing.
- Prefer tables, short bullets and headed blocks over long prose, and keep prose for making an
  argument.
- Name every number's source table or figure, so a reader can go and look.
- A table cell holds one line. Markdown has no lists inside cells, and the conversion to PDF
  silently drops the workarounds.
- Maths goes in a fenced block, never in a table cell.
- Keep it short: leave out anything a reader does not need to act on or check.

**Say nothing about the page.** The report renders portrait by default, and the build gives a
rotated page to any figure it measured as too wide for a portrait column. Never write an image
width, a page break or any LaTeX.

**Length.** A complete report runs to about **34 to 37 pages of body** and 53 to 55 including the
appendices. Most of that is components, tables and figures rather than prose you write, so match
the shape instead of counting words: every section below, each with the components it calls for, the figures
it names and one Analysis Note. If a section is running long, the full table belongs in an appendix
and a summary belongs in the body.

---

## Sections

Each heading is fixed. Each entry says what the section must contain.

### Title block

`# SeqMaestro Report: <dataset and task, in the reader's words>`

Under it: what the target is, and the date the analysis was completed. For a binary target that is
the two classes, in one line each. For a continuous one it is what the target measures and what a
higher value means, in one line.
No run directory, no commit, no internal identifier.

### Summary

Half a page. Three or four sentences and one short table.

- What the question was.
- The best few-feature model, what it scores on held-out sequences, and what the base composition of
  the sequence already achieves.
- The two or three ideas that held up across independent configurations.
- The single caveat that most limits what the study can say. One, not a list.

Everything here restates something below. Add a pointer to the glossary.

### How to read this report

Include `components/00_how_to_read.md` verbatim.

### Stage 1. The data and its preparation

This is the section the readers are best placed to check, so write it in their terms and in order.

- **What the target is**, in the biology, one paragraph. For a binary target, what the two classes
  are: say plainly whether both classes come from one experiment, and what the negative class is.
  For a continuous target (`facts.target`), what it measures, in which units, what a higher value
  means, and how its values are spread (each split's summary in `stage1.splits`): where most sit, and
  how far the tails run.
- **What each sequence is**: window, what it is anchored on, what is included and what is left out.
  Where the sequences differ in length, give their range (`stage1.sequences.observed_length`), and
  say which features use positions in the positional window (`stage1.sequences.positional_window`:
  its length, the end it is anchored to, and whether a shorter sequence was padded or filtered out).
- **The measurements supplied with each sequence**, where the dataset has covariates
  (`stage1.covariates`): what each one measures, in the declaration's words, and its range on the
  training rows.
- **The preparation, as ordered steps**, each with the rows it dropped. A step that dropped none says
  so.
- **The split**: sizes, the class balance in each (or a continuous target's spread in each), what
  the split keeps together and why.
- **What the models were told.** First, whether the run was blinded (`stage1.blinding.blinded`).
  If it was, name the fact the language models were not told (`stage1.blinding.withheld`) in plain
  words, then include `components/22_blinding.md` verbatim. If it was not, say whether the task
  description named the entity under study. If it did, say so once, plainly, and say that Stage 3B's
  held-out test does not depend on whether a feature was recalled or discovered. Do not raise either
  again outside this item and the final limitations. Then say whether the person running the
  analysis gave guidance on the form of the features (`stage1.feature_guidance`): quote it in full,
  or say that there was none.
- **What a model gets for free**: the base composition of the sequence (`tables/baseline.md`, and
  `facts.json` under `baseline`), and the detection floor of the held-out split (`stage1.design_power`).
  Include the baseline table wherever you first need it, here or in Stage 2, and not twice.
- **What Stage 1 put to the user** (`stage1.findings`): each finding, what was found, and the answer
  the declaration records with its reason. A finding with no answer yet is an open question and goes
  in the next item.
- **Open questions for you.** A short list of things a clarification from the collaborators would
  settle, each with what changes depending on the answer. Where a question would invert every
  direction in the report, put it first and say so.

> **AI Agent Analysis Note**

### Stage 2. Which representations and model classes capture the signal

- Include `components/20_feature_sources.md` verbatim.
- When the dataset supplies covariates, include `components/21_covariates.md` verbatim.
- When a configuration's feature sources include motifs, include `components/23_motifs.md` verbatim.
- The abbreviation key: `tables/key.md`.
- Include `components/33_model_reference.md` verbatim.
- Include `components/10_evaluation.md` verbatim.
- `{{figure:stage2_grid}}`.
- The **headline set**, `tables/stage2_headline.md`, in the body: the best configuration of each interpretable model class,
  the best one built on enumerated features, and the strongest reference model. `facts.json` records
  which configurations those are, under `headline_set_stage2`. Its `headline_set` is the same
  choice made among the reduced models, which Stage 3A reports. The full grid,
  `tables/stage2_all_arms.md`, goes in Appendix A.
- Compare every configuration with the base-composition line, and say how many fail to clear it.
- **Models compared on one written source** (the sources the report calls "written" and "written,
  iterative") were each given a separate proposal. Wherever the report compares them, say so: a difference between them includes
  the difference between their proposals.
- **Does a configuration find the same things twice.** Include
  `components/60_stage2_reproducibility.md` verbatim, and `components/61_stage2_judge.md` after it
  where the run's config ran the judge (`facts.json`: `stage2.judged`), then
  `tables/stage2_reproducibility.md`, which has a row for every configuration.

> **AI Agent Analysis Note**

### Stage 3A. Models small enough to interpret

- Include `components/40_few_feature_models.md` verbatim.
- `{{figure:stage3a_grid}}` and `{{figure:stage3a_cost}}`, and table `tables/stage3a_all_arms.md`.
- **The reference line.** Every configuration here is allowed the same small number of features, and
  the grid figure shows the base-composition line. `facts.json` records under `baseline` whether
  the base composition fits the same budget (`within_budget`). Say which configurations clear it, and
  whether the comparison is like for like.
- **One model of each interpretable kind the run has.** For each: the component explaining how that
  model class works, then the figure, then the table of its features.
  - additive: `components/30_model_ebm.md`, `{{figure:stage3a_ebm}}`, `tables/features_<arm>.md`
  - tree: `components/31_model_tree.md`, `{{figure:stage3a_tree}}`, `tables/features_<arm>.md`
  - sparse linear: `components/32_model_l1.md`, `{{figure:stage3a_l1}}`, `tables/features_<arm>.md`
  - IGSR, on a continuous target: `components/34_model_igsr.md`, `{{figure:stage3a_igsr}}`,
    `tables/features_<arm>.md`
  - symbolic: `components/35_model_pysr.md`, then `{{table:stage3a_pysr_equation}}` (the equation,
    one term to a line, and what each of its features measures), `{{figure:stage3a_pysr}}`,
    `{{figure:stage3a_pysr_front}}`, `tables/features_<arm>.md`. Say which features the equation
    combines and how, in words, and whether its `metrics.json` counts rows it was undefined on
    (`filled_rows`).
  - After each figure, two or three sentences on what that specific model says. Name the features it
    leans on and what they measure.
  - **A model that uses no feature** predicts one value for every sequence. Its file under
    `internals/` has `kind: no_features` with that value, and the build writes no figure and no
    equation for it. Where the arm's Stage 2 model already used none, `facts.json` gives its
    `n_final` as 0 and there is no table of features. Where the reduced model gives no weight to
    the columns that survived (an equation that is a constant), `n_final` counts those columns
    and their table is written, although the model uses none of them. A tree with no split is
    exported as a tree of one leaf and is drawn as one box. The subsection keeps the component,
    then states in a sentence or two that the model uses no feature and what it scores.
- **A configuration whose features were written with validation in view.** The iterative source
  and IGSR choose on the validation set while they write, and the reduction starts from the seed
  that scored best on it. Say so where such a configuration is shown: its validation scores, and
  the choice of its seed, are partly the result of that steering.
- **What survives in several configurations**, by feature name: `tables/stage3a_shared_features.md`
  lists the features at least `{{param:shared_at_least}}` configurations kept. Say that matching on
  the name is conservative, and that Stage 3B answers the question properly.

> **AI Agent Analysis Note**

### Stage 3B. Which features would survive a repeat

- Include `components/50_reliability_overview.md` verbatim.
- `tables/stage3b_summary.md`, with a sentence on what the columns are counting.
- Include `components/51_tier1_stability.md` verbatim, then the tier-1 results: the "chosen again"
  counts of `tables/stage3b_summary.md`, and `stage3b.arms` in `facts.json` for a number in prose.
  Where a configuration's `pool_caveat` there is set, say in a sentence what it means for that
  configuration's tier-1 results: its candidates were the tree's output, or include motifs found
  with the labels of the sequences Tier 1 resamples, and either makes its frequencies too high.
- Include `components/54_thresholds.md` verbatim, then `{{figure:stage3b_thresholds}}`.
- Include `components/52_tier2_association.md` verbatim, then `tables/stage3b_tier2.md`. The direct
  plug-in check has a separate table, `tables/stage3b_plug_in_fdr.md`, which goes in Appendix B.
- Include `components/53_tier3_necessity.md` verbatim, then the tier-3 results: the "model needs
  it" counts of the same summary table.
- `{{figure:stage3b_evidence}}`, which covers the configurations the body shows. Per-configuration
  tables (`tables/evidence_<arm>.md`) go in Appendix B, with `{{figure:stage3b_evidence_all}}`. Put
  at most one evidence table in the body, as a worked example.
- Include `components/55_concepts.md` verbatim, then `{{figure:stage3b_concepts}}`,
  `tables/concepts.md` and `tables/concepts_detail.md`.
- **The checks.** In a short paragraph, say that the association test was run again with the
  labels shuffled for every configuration, and the stability resampling for one, that these called
  nothing, and that a simulated signal planted in real columns was recovered in the ranking. Say
  what the planted-signal check showed about the study's power.
- **Configurations that were not profiled.** Say how many of the grid were profiled and why the rest
  were not. `facts.json`'s `stage3b.not_profiled` records the reasons in our terms rather than the
  reader's: a black-box model, or a configuration left out for its cost. Paraphrase them into what
  they mean for the reader, which is that the configuration is absent rather than that it failed.

> **AI Agent Analysis Note**

### What this suggests looking at next

**This whole section is one AI Agent Analysis Note and is headed as one.** Keep it brief, under a page.

Two separate lists, because the model to use and the ideas to pursue serve different goals.

- **The model to use**, if the goal is to score new sequences. Name it, say what it needs and what it
  achieves, and say what it would take to trust it on sequences from a different source.
- **The ideas to pursue**, if the goal is a follow-up experiment. Rank them by weight of evidence
  across the stages. For each: what it measures, how many independent configurations reached it,
  whether it held on held-out sequences, and which direction it points. Stop there. Do not design
  an assay.

Where the two lists disagree, say so plainly and say why they can disagree.

### What this analysis cannot tell you

Keep it short: the three or four points that most limit what a reader can do with the report, and
nothing else, each in two or three sentences.

Candidates, where they apply: the size of the held-out split and what it can resolve; anything the
design confounds that no control here can catch; a preparation decision that is unresolved; the
scope the sequences cover. Leave out anything already covered by a number in the body.

> **AI Agent Analysis Note**

### The language models in this analysis

Short: two tables, with a sentence before each saying what it lists.

- `tables/language_models.md`: the language model the analysis called, the reasoning effort each
  call asked for, and what each part of the analysis called it for (writing the features of the
  written sources, rating them where the judge ran, and grouping them into ideas). A run calls one
  model under one setting, so the table has one row. If it has more, say that the model or how it
  was called changed part way through the run, and which parts each row served.
- `tables/agents.md`: the agent that carried out the analysis with the user and wrote this report,
  and the model it runs on, each as it recorded itself. Once you have recorded yourself for the
  report (`process/reporting.md`), it names you.

### Appendix A. Every configuration

`tables/stage2_all_arms.md`, `tables/stage3a_all_arms.md` and `tables/stage2_cross_arm_concepts.md`.

### Appendix B. Every reported feature with its evidence

One or two sentences of framing, then `{{figure:stage3b_evidence_all}}`, then
`tables/stage3b_plug_in_fdr.md`, then one `tables/features_<arm>.md` and one
`tables/evidence_<arm>.md` per profiled configuration, under a heading naming the configuration.
Close with the reference model's `tables/features_<arm>.md`, and say that it is not interpreted.

**Write the framing sentences even though the heading says enough.** That figure is often wide
enough to be given a rotated page, and a rotated figure directly under a heading leaves that heading
nearly alone on a page of its own.

### Appendix C. Glossary

Include `components/90_glossary.md` verbatim.

### Appendix D. Where the numbers come from

A short table: which file each section's numbers were read from, so a bioinformatician can go and
look. `facts.json` lists them for every configuration, under `files`. Give paths relative to the
run directory, never absolute ones.

---

## Before you hand it over

- Every number traced to a table or a figure you included. Nothing retyped, nothing recomputed.
- Every feature name copied, never typed.
- Every reported feature has a direction, or a stated reason it has none.
- The table of agents names you, for the report.
- For a binary target, the class balance is stated, and the metrics quoted are the ones that
  balance allows. For a continuous target, the target's spread is stated, and wherever a few extreme
  values could drive R², Spearman's rho is quoted beside it.
- Stage 3A names the reference line, says whether it fits the same feature budget, and says which
  configurations clear it.
- Every analysis has its explanation before its numbers.
- No internal vocabulary, no directory names, no discussion of our tooling.
- Table rows all have the same column count, and the PDF builds and was looked at.
