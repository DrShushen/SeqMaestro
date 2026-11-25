# Recurrence

> This page tracks `seqmaestro/pipeline/analyses/recurrence.py`, which `analyse_recurrence.py` calls. Change it in the same commit as that module.

The [recurrence analysis](../../glossary.md#recurrence-analysis) answers two questions about the arms whose features a language model writes, those on a [written source](../../glossary.md#written-source):

- Does an [arm](../../glossary.md#arm) keep arriving at the same ideas from one [seed](../../glossary.md#seed) to the next?
- Do different arms arrive at the same idea, when each is free to name it and to place its window differently?

An *idea* is a group of features that measure the same underlying property. A language model groups the written features into ideas, and the analysis counts how often each idea recurs. The tables and the code call an idea a *concept*, whereas a [concept](../../glossary.md#concept) elsewhere in SeqMaestro is a group of features whose values agree, so this page calls the language model's groups ideas.

## Ideas under new names

The stability analysis ([stability and co-discovery](stability_and_co_discovery.md)) measures whether an arm's [top N](index.md#the-top-n) features are the same set again at another seed. That is the right question for an arm that chooses from a fixed vocabulary, which is the same at every seed, so that selecting the same members again means something.

For an arm that writes its features, the question fails in a specific way. Such an arm gives the same idea a new name at every seed and computes it over a slightly different window, and neither of the matcher's two ways of matching ([matching two sets of features](index.md#matching-two-sets-of-features)) sees through that:

- Pairing by definition compares the features' code, with names and formatting normalised away, and two functions that compute the GC content over positions 0 to 29 and over positions 5 to 34 are different code.
- Matching by values correlates the columns of values, and windows that overlap only in part do not follow each other closely enough to reach the threshold of 0.8.

An arm on a written source can therefore propose the same idea at every seed, under a different name each time, and score near zero on both. The recurrence analysis measures the arm's repertoire of proposals, whereas stability's `top_n_kappa` measures what the arm selects. An arm can be stable in one and not the other, as a tree-building arm typically is: it proposes a small, consistent repertoire and then splits on enumerated positional features at most nodes.

## The features grouped

- **The arms** are every arm of the run whose pool holds a written source: the one-round source (`llm_generated`), the [iterative source](../../glossary.md#iterative-source) (`llm_generated_iterative`) or the [joint source](../../glossary.md#joint-source) of DEFT and IGSR (`llm_generated_joint`). On each of them, a language model writes a cell's features for that cell alone, so one idea can come back under another name at the next seed. This holds for the one-round source too: its proposal is fixed before the fit, which is why its features are marked `selected`, but each cell still has a proposal of its own.
- **They are found** from the run's directories and the names of their pools ({py:func}`~seqmaestro.pipeline.analyses.recurrence.written_arms`), so an arm that a config adds is included without a list kept in the script. `--arms` groups the arms it names instead. A run with no such arm has nothing to group, and the analysis says so and writes nothing.
- **The features** are those of the kinds in `analyses.recurrence.kinds`, which the shared config sets to the written kind (`llm_generated`). An enumerated positional feature is drawn from a fixed list, so its recurrence would follow from arithmetic alone.
- **How many**: each cell's top N by the model's importance, among its features of the kinds grouped. On a pool that also holds other sources, a written feature outside the cell's overall top N can therefore be grouped, whereas stability and co-discovery would leave it out.
- **Both label conditions** are grouped together, which is the first of the safeguards below.

## The grouping

A language model is given a numbered list of feature names and descriptions, and asked to group them by the underlying property each one measures. Its instructions are `SYSTEM` in [`recurrence.py`](../../../seqmaestro/pipeline/analyses/recurrence.py), followed by the dataset's [task description](../../glossary.md#task-description) as the models that write features are told it.

The instructions name no domain. What counts as the same idea in a dataset comes from its task description, which the [proposer](../../glossary.md#proposer) and the [judge](../../glossary.md#judge) are told too. Naming one dataset's motifs in the script would make the analysis correct for that dataset and quietly wrong for the next one. The rules the model is given therefore concern granularity alone: for example, the same quantity measured over different windows is one idea, whereas different mechanisms stay apart.

The granularity is set in the prompt, at roughly one group for every few items (`analyses.recurrence.items_per_group` in the config), for two reasons:

- It has to be the same for both label conditions, since otherwise a comparison between them would compare nothing.
- A model left to choose returns nearly the identity: almost as many groups as features, with one motif split into its two strand forms.

The clustering of co-discovery cannot be used instead, because it decides sameness by the features' values, which do not see one idea computed over two windows. The two groupings also differ in what they group and what they need:

| | Co-discovery | Recurrence |
|---|---|---|
| Groups | Every arm's features at one seed | One arm's features at all its seeds, and in the pass across arms, every arm's features at all seeds |
| Two features are the same when | Their values correlate on the reference rows | A language model judges from their names and descriptions that they measure the same property |
| Needs | Every arm evaluated on the same rows | Nothing shared but the task description |
| Answers | Did another arm find this property? | Did this arm think of it again, and did another arm think of it? |

## Safeguards

The counts mean something only with all three of these safeguards:

1. **The permuted cells are in the same grouping pass, and hidden in the same way.** Ideas are defined once over both label conditions and then counted separately for each. Grouping the two conditions separately would produce two different sets of ideas and no comparison at all, and defining ideas on the real features alone would let the grouping fit itself to what the real cells happened to produce.
2. **The model never sees the seed, the label condition or the arm.** The features arrive shuffled, under numbers that reveal nothing, and the key from each number to its feature is written separately, so that the grouping can be checked against what the model saw.
3. **The permuted count is reported beside every real count.** A language model that writes features from the task description proposes much the same features on shuffled labels as on real ones. An arm that keeps many more features on real labels than on shuffled ones has therefore already shown that the labels steer what it keeps, before any idea is counted. The analysis prints the number of features under each label condition first, and warns when an arm has no permuted features.

## The tables

The analysis writes three tables under the run's `results/`, the first only where the pass across arms runs. They differ in the grouping pass that defines their ideas:

| File | Grouping pass | What it counts |
|---|---|---|
| `cross_arm_recurrence.csv` | One pass over every arm at once | Which arms reached each idea, and at how many seeds |
| `concept_recurrence.csv` | One pass for each arm | The ideas an arm returned to across its seeds |
| `feature_recurrence.csv` | None | The features that recur by name, character for character, across seeds |

`concept_recurrence.json` records each pass: its counts, the key from each listed number to its feature, and the language model that grouped them.

**The scope of a grouping pass defines what an idea means.** An idea in `concept_recurrence.csv` is defined inside one arm's pass. When two labels from two arms look alike, the grouping model has chosen the same words twice, which is no evidence that their members match. Comparing those labels between arms is therefore a mistake. `cross_arm_recurrence.csv` makes the comparison properly: every arm's features enter a single pass, so an idea is one object across the arms, and `real_arms` counts how many arms reached it.

**The count by name and the count by idea are both written**, because the count by name misdescribes written features. A written feature recurs by name only if it was named in the same way again, however consistent its arm's ideas are, whereas an enumerated feature recurs by name by construction, since its arm selects from the same list of names every time. `feature_recurrence.csv` gives each feature's `provenance` ([the feature record](../feature_record.md)): a written feature is `selected` where its proposal was fixed before the fit, and `generated` where it was written during the fit. A feature of an enumerated kind, which `analyses.recurrence.kinds` can add, is `selected` too, so its rows must not be averaged with the written ones.

## The columns to quote

Each table of ideas has one row per idea, with how many seeds and features of each label condition it holds. {py:func}`~seqmaestro.pipeline.analyses.recurrence.summarise` lists the columns.

**`p_enriched` is the column to quote, and the coverage of seeds (`real_seeds`) is the one that misleads.** Coverage measures whether an idea turns up again across seeds, whereas enrichment measures whether the labels produced it: `p_enriched` is a [hypergeometric](https://en.wikipedia.org/wiki/Hypergeometric_distribution) test of whether the idea holds more real features than drawing its members blind would give. An arm that keeps several times as many features on real labels reaches every real seed in any large idea, so coverage alone says little.

`p_chance` exists for the same reason. {py:func}`~seqmaestro.pipeline.analyses.recurrence.recurrence_null` shuffles which features have which tag of label condition and seed, keeping the size of each idea and the number of features of each label condition fixed, and counts how often chance alone reaches the observed number of real seeds. An arm with more real features than permuted ones would otherwise show more real seeds per idea for that reason alone, whatever its ideas are.

## Running the analysis

The [driver](../../glossary.md#driver) runs the analysis after Stage 2. It makes one call to the language model for each arm, and one more for the pass across arms, which a config leaves out with `analyses.recurrence.across_arms: false`. A real run needs the language model's settings in the environment (`set -a && source .env && set +a`), whereas `--dry-run` prints the prompts and makes no call. Every call is made under the config's `language_model` settings, and `concept_recurrence.json` records which model grouped the features and how it was called (`language_model`).

A grouping is one answer of a language model, so the analysis requests it once. It keeps the tables it wrote while what they were computed from is unchanged (the features, the settings, the task description, the model and its reasoning effort), and groups again only with `--force` ({py:mod}`seqmaestro.pipeline.analyses.inputs`). The setting `analyses.recurrence.seed` fixes the order in which the features are listed and the draws behind `p_chance`.

A reply names every listed feature by its number, so its length grows with the list. Each call is therefore given the tokens and the time that a reply to its list needs ({py:func}`~seqmaestro.pipeline.analyses.recurrence.reply_tokens` and {py:func}`~seqmaestro.pipeline.analyses.recurrence.reply_seconds`). One limit for every call would truncate the reply of the pass across arms, and a truncated reply cannot be parsed, which fails the analysis.

## Limits

**The grouping is a language model's judgement.** The model does not see the seed or the label condition, and the order of its list is seeded, yet a second call on the same features can group them differently. A table of ideas is therefore a structured account of the repertoire, which cannot be rederived exactly.

**The pass across arms is the call most likely to leave features ungrouped**, since it puts every arm's features in one list. In `concept_recurrence.json`, `n_grouped` is the number of listed features that the grouping placed in at least one group, to compare with `n_features`. The log notes how many features the grouping left out and how many it placed in more than one group, and a feature in two groups counts towards both ideas.

**It says nothing about whether an idea is right.** An idea that an arm reaches at every seed, and significantly more often on real labels than on shuffled ones, is one the arm holds consistently and that the labels influenced. Whether it holds is a separate question: [direct measurement](direct_measurement.md) tests what one feature does in the data, and [the judge](judge.md) rates how much it would matter if it held.
