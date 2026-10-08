# Recurrence: does an arm arrive at the same idea again?

> **Maintained page.** Change it in the same commit as
> `seqmaestro/pipeline/analyses/recurrence.py`, which `analyse_recurrence.py` calls.

`analyse_recurrence.py` answers two questions the other analyses cannot. When a language model
**writes an arm's features**, does the arm keep thinking of the same things across seeds? And do
**different arms** think of the same thing, when each is free to name and window it differently?

This differs from stability in four ways: it needs a different notion of sameness, a different
clustering, a language model in the loop, and its own null. This page says why each is needed.

---

## 1. Why stability cannot answer it

`analyse_stability.py` asks whether an arm's reported top 20 is the same *set* again. That is the
right question for an arm choosing from a fixed vocabulary: the vocabulary is the same every seed,
so selecting the same members means something.

It is the wrong question for an arm that invents its features, and it fails in a specific way. A
generative arm renames the same idea every time and instantiates it over a slightly different
window. **Neither pass of the matcher sees through that:**

- *Pairing by definition* hashes the normalised source, and two functions computing GC over
  positions 0 to 29 and 5 to 34 are different strings.
- *Behavioural matching* correlates the value columns, and partially overlapping windows on opposite
  flanks do not track each other well enough to clear the 0.8 threshold.

So a generative arm can propose the same idea in every seed, under a different name each time, and
score near zero on both passes.

**This analysis therefore measures the arm's *proposal* repertoire**, where `top_n_kappa` measures
its *selection*. An arm can be stable in one and not the other. A tree-building arm typically is:
it proposes a small consistent repertoire and then selects naive positional splits at most nodes.

---

## 2. Three tables

| file | grouping pass | counts |
|---|---|---|
| `cross_arm_recurrence.csv` | **one over every arm at once** | which **arms** reached each idea, and in how many seeds |
| `concept_recurrence.csv` | one per arm | the **ideas** an arm returned to across its seeds |
| `feature_recurrence.csv` | none | features that recur **character for character** across seeds |

**The scope of a grouping pass is what a concept means.** A concept in `concept_recurrence.csv` is
defined inside one arm's pass. So when two labels from two arms read alike, the grouping model has
chosen the same words twice, and that is no evidence that their members match. Comparing those
labels between arms is the mistake this page most wants to prevent. `cross_arm_recurrence.csv` is
the output that makes the comparison properly: every arm's features enter a single pass, so a
concept is one object and `real_arms` counts how many arms are in it.

The by-name and by-idea tables are both always produced, because which one describes an arm depends
on whether a language model writes that arm's features:

- An arm choosing from a fixed vocabulary passes the by-name count **by construction**, because it
  selects from the same list of names every time. An arm whose features are written fails it
  however consistent its ideas are.
- The by-idea count is the reverse.

The two disagree most where it matters, and quoting either alone misdescribes half the arms. The
cross-arm table asks a third question: whether any other arm arrived at the same place.

`feature_recurrence.csv` states each feature's `provenance` on its row
([`feature_record.md`](../feature_record.md)). A written feature recurs by name only if it was
named the same way again, whether its proposal was fixed before the fit (`selected`) or it was
written during the fit (`generated`). Where a config's `analyses.recurrence.kinds` also takes an
enumerated kind, those features are `selected` too and recur by name by construction, so their
rows are not to be averaged with the written ones.

---

## 3. What gets fed in

**Which arms.** Every arm of the run whose pool holds a written source: the one-round source
`llm_generated`, the iterative source `llm_generated_iterative`, or the joint source
`llm_generated_joint` of DEFT and IGSR. On each of them a language model writes a cell's features
for that cell alone, so one idea can come back under another name at the next seed. That holds for
the one-round source as for the others: its proposal is fixed before the fit, which is why its
features are stamped `selected`, and each cell still has a proposal of its own. The arms are found
from the run's directories and each pool's name, so an arm a config adds is taken without a list
kept in the script. A run with no such arm has nothing to group, and the analysis says so and
writes nothing. `--arms` groups the arms it names in their place.

**Which features.** `llm_generated` kinds only, by default. A naive positional split is drawn from an
enumeration, so its recurrence is arithmetic and not evidence.

**How many.** Each cell's top 20 by the model's importance among its features of the kinds asked
for. On a pool that also holds other sources, a written feature outside the cell's overall top 20
is therefore grouped, which stability and co-discovery would leave out.

**Both label conditions**, together. This is the first of the three safeguards in §5.

---

## 4. The grouping

A language model is given a numbered list of feature names and descriptions and asked to group them
by the underlying property each one measures.

The instructions are **domain-free**. What counts as "the same idea" in a given dataset comes from
that dataset's own description, which is appended to the prompt and is the same description the
proposer and the judge read. Naming one dataset's motifs in the script would make the analysis
correct for that dataset and quietly wrong on the next one.

The rules given are about granularity, not about biology:

- The same quantity over different window widths, offsets or regions is **one** group.
- Presence, count, density, fraction and difference-of-counts of the same thing are **one** group.
- The same pattern read under a symmetry of the representation is **one** group, where the dataset
  description implies the two readings describe one process.
- Genuinely different mechanisms stay apart. A feature indexing a single arbitrary coordinate, with
  no stated meaning beyond its position, is its own group.

Granularity is **set in the prompt** (roughly one group per *N* items, `analyses.recurrence.items_per_group` in the run's config) for two
reasons. It has to be identical for both label conditions, or the comparison compares nothing. And a
model left to choose returns close to the identity: nearly as many groups as features, with one
motif split into its two strand readings.

### Why not the co-discovery clustering?

The two clusterings run along different axes:

| | co-discovery | recurrence |
|---|---|---|
| clusters across | **arms**, within one seed | **seeds**, within one arm |
| sameness decided by | how feature values correlate on shared reference rows | what the name and description say it computes |
| needs | every arm evaluated on identical rows | nothing shared but the dataset description |
| answers | "did anything else find this?" | "did this arm think of it again?" |

Behaviour cannot answer the second, for the reason in §1: the same idea over a shifted window does
not correlate with itself.

---

## 5. The three things that keep it honest

The output means nothing without all three.

**The permuted arm is in the same clustering pass, blinded identically.** Concepts are defined once
over both conditions and then counted separately. Clustering the two conditions separately would
produce two different concept sets and no comparison at all. Defining concepts on the real features
alone would let the grouping be fitted to what the real arm happened to produce.

**The model never sees the seed or the label condition.** Features arrive shuffled under opaque
numbers. The key is written separately, so the clustering can be inspected on the same terms the
model saw it.

**The permuted count is reported beside every real count.** An arm that reads a task description
and not the labels proposes the same repertoire either way, and the language-model proposer is such
an arm. An arm that keeps many more features on real labels than on shuffled ones has already
said something before any concept is counted. That is why that line prints first, and why a missing
permuted set raises a warning.

---

## 6. What comes out, and which column to read

One row per concept:

| column | |
|---|---|
| `real_seeds`, `permuted_seeds` | how many seeds of each condition contain this concept |
| `real_features`, `permuted_features` | how many features of each fell into it |
| `p_enriched` | hypergeometric: given how many features each condition contributed overall, is this concept more real-heavy than drawing its members blind would give? |
| `p_chance` | the permutation null on coverage: how often chance alone reaches this many real seeds, given the concept's size |
| `example` | one member's name, so the row is readable |

**`p_enriched` is the column worth reading, and seed coverage is the one that misleads.** Coverage
asks whether a concept turns up again across reseeding. Enrichment asks whether *the labels produced
it*. An arm keeping several times as many features on real labels reaches every real seed in any
large concept, so coverage alone says little.

`p_chance` comes from `recurrence_null`: shuffle which features carry which `(condition, seed)` tag,
holding the group sizes and the per-condition counts fixed, and ask how often chance alone reaches
the observed number of real seeds. It exists for the same reason: an arm with more real features
than permuted ones shows more real seeds per concept for that reason alone, whatever the concepts
are.

---

## 7. What it cannot support

**The grouping is a model's judgement, not a measurement.** It is blinded, and `--seed` fixes the
order the features are listed in and the draws of the null. The grouping itself is one answer of a
language model, and a second call on the same features can group them differently. The analysis
therefore keeps the tables it wrote while the features, the settings and the description are
unchanged, and groups again only with `--force`. Read a
concept table as a structured reading of the repertoire, not as a partition anyone could rederive.

**It needs credentials and costs a call per arm, plus one more for the cross-arm pass.** Run
`set -a && source .env && set +a` before a real run. `--dry-run` prints the prompt and makes no
calls, and a config with `analyses.recurrence.across_arms: false` leaves the extra pass out.

**The cross-arm pass puts every arm's features in one listing**, so it is the largest call the script
makes and the one most likely to leave features ungrouped. `n_grouped` in the JSON is the number
of listed features the grouping placed in at least one group, to set against `n_features`. The log
notes how many it left out and how many it placed in more than one group. A feature in two groups
counts towards both concepts.

**A reply names every listed feature by its number**, so its length follows the listing's. Each
call is therefore given room and time for a reply of its listing's length (`reply_tokens` and
`reply_seconds` in the module). One limit for every call would cut the cross-arm reply short, and
a reply cut short cannot be parsed, which fails the analysis.

**It says nothing about whether an idea is *right*.** A concept an arm reaches in every seed, and
significantly more on real labels than shuffled ones, is an idea the arm holds consistently and that
the labels influenced. Whether it is true of the biology is what the judge and the direct-effect test
are for.
