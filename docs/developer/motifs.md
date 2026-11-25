# Motifs

> This page tracks the motifs package ({py:mod}`seqmaestro.motifs`), the motif featurisers ({py:mod}`seqmaestro.features.motif`), the step that prepares a run's motifs ({py:mod}`seqmaestro.pipeline.motifs`) and the motif sources' settings in [`_motifs.yaml`](../../experiments/configs/_motifs.yaml). Change it in the same commit as any of them.

A motif describes the sites that a protein binds, such as a transcription factor on DNA or an RNA-binding protein on RNA. It is held as a position frequency matrix: one row per position of a site, with how often each base occurs there. A motif feature summarises how strongly, or how often, one motif matches anywhere in a sequence. SeqMaestro has two motif [feature sources](../glossary.md#feature-source) ({py:mod}`seqmaestro.features.motif`):

- **Known motifs** (`motif_known`): every motif of a published collection, such as [JASPAR](https://jaspar.elixir.no/)'s for the transcription factors of one taxon, or [CisBP-RNA](https://cisbp-rna.ccbr.utoronto.ca/)'s for the RNA-binding proteins of one species.
- **Discovered motifs** (`motif_discovered`): the motifs that [STREME](https://meme-suite.org/meme/doc/streme.html), from the [MEME Suite](https://meme-suite.org/meme/), finds in the rows a [cell](../glossary.md#cell) fits on, by comparing its two classes or the two ends of its target ([discovered motifs](#discovered-motifs)).

Since both sources match motifs to sequences in the same way, every motif feature has the kind `pwm_scan` ([the feature record](feature_record.md)).

## Matching a motif to a sequence

{py:mod}`seqmaestro.motifs.scan` describes how a window of a sequence is scored against a motif, how each score is given an exact p-value, and why scans made on different machines agree. One of its choices matters for interpreting a feature: **the background is uniform**, a quarter for each letter, as JASPAR's files state. A scan then depends on the sequence and the motif alone, so a run can scan each sequence once and share the scan. On a dataset of skewed composition, however, a motif rich in the commoner letters matches by chance more often than its p-value says. The composition [baseline](../glossary.md#baseline), which every arm is compared with, shows how far composition alone predicts the target.

**Which strands are scanned is a fact about the dataset**, so the dataset's config states it: `both` for DNA, which a protein can bind on either strand, and `given` for RNA, which is single-stranded.

## Best match and count

Each motif gives up to two columns, which {py:mod}`seqmaestro.motifs.scan` defines: `best_match`, the strength of the best match anywhere in the sequence as `-log10` of its p-value, and `count`, the number of positions whose p-value is at or below a threshold. Three choices shape what the two columns mean:

- **The best match is given as a p-value**, because a p-value means the same for a motif of six positions as for one of twenty, whereas a raw score does not. A threshold that a tree chooses, or the steep part of an additive model's curve, can then be stated as "a match at p below 1e-5".
- **Whether a motif matches at all** is `count > 0`, which a tree or an additive model finds by itself, so it has no column of its own.
- **On sequences of different lengths**, a count grows with the length, which `normalise: per_length` corrects by dividing each count by the positions scanned, as it does for the k-mer counts. The best match also rises with the length, because a longer sequence offers more windows for a chance match, and no normalisation removes this rise.

**Only the choice of count columns is learned at fit.** A count is kept where enough of the fitting rows hold a match, by the rule the [reliability panel](../glossary.md#reliability-panel) applies first, so that no model is offered a count that the panel would discard ({py:mod}`seqmaestro.features.motif` gives the rule). Every best match is kept. The featuriser's description, which each cell records, states how many count columns were dropped, how many motifs can never reach the threshold, and how many fitting rows could not be scanned in full.

**Each sequence is scanned once per run.** Every cell of every arm transforms the same training and validation sequences, as do the reduction and the panel, so a run keeps its scans in a store ({py:mod}`seqmaestro.motifs.store`), and a featuriser scans only the sequences that the store lacks.

## Known motifs

Each known motif is the binding preference that an experiment measured for a protein, and the motif is named after that protein. A model that relies on `motif_MA0139_2_CTCF_count` says that the target goes with the motif of CTCF, which a biologist can act on. The motifs also bring the literature into the run. An arm on known motifs can find only what the collection already holds, and a reader can recall what is known about each name, as a language model can. The k-mer and positional sources hold no such knowledge.

**The collection** is fetched once per machine at Stage 1 by [`fetch_motif_collection.py`](../../experiments/prepare/fetch_motif_collection.py), with a record of where it came from and under which licence ({py:mod}`seqmaestro.motifs.published`). A run copies the collection's motif file into its own directory the first time it scans with it, and every process of the run uses the copy ({py:mod}`seqmaestro.pipeline.motifs`). A collection fetched again while the run goes on therefore changes nothing in the run, and a finished run keeps the motifs it scanned with.

**Blinding.** In a blinded run, a pool that holds the known motifs and a [written source](../glossary.md#written-source) is refused when the config is loaded. A known motif is named after its protein, and a language model that writes features beside it can be shown the names of the columns a fit relied on, as the [iterative source](../glossary.md#iterative-source) is. It would then be shown the names of proteins, which [blinding](../glossary.md#blinding) withholds. A discovered motif is named by its consensus, so it is allowed beside a written source.

## Discovered motifs

STREME finds the motifs that are more common in one set of sequences than in another ({py:mod}`seqmaestro.motifs.discovery`). For a binary target, it compares the fitting rows of class 1 with those of class 0. For a continuous target, it compares the rows at the top of the target with those at the bottom, because STREME takes no score or weight for a sequence, and the rows between them take no part. Where the config's `directions` is `both`, it searches both ways, since a motif enriched in either set can predict the target. Each motif found gives the same columns as a known motif, under a name that says how it was found: `streme_up_1_GATTACA_best` is the best match of the motif that STREME ranked first among those enriched in class 1, or at the top of the target, with the consensus `GATTACA`.

**The rows and labels are the cell's**, so the motifs of a [permuted](../glossary.md#label-condition) cell are found on shuffled labels, just as the [proposer](../glossary.md#proposer) writes features for shuffled labels. The control therefore remains a control for the whole arm, discovery included. Every cell at one label condition and seed fits on the same rows with the same labels, so a run makes each discovery once and keeps it in its store ({py:mod}`seqmaestro.pipeline.motifs`). The exception is `permutation_scope: cell`, under which a control's shuffled labels depend on its arm, so each control cell makes its own discovery.

**The closest known motif.** Where the run names a collection to compare with (`annotate_with`), [Tomtom](https://meme-suite.org/meme/doc/tomtom.html) finds the known motif closest to each motif found. It is kept in the record's `extra` (`closest_known`) for a reader of the report, which names it beside the feature where the two are alike. It appears in no feature's name or description, so a language model shown the features is never shown a protein's name through it.

**Tier 1 of the reliability panel is optimistic for a discovered motif.** Tier 1 resamples the training and validation rows, and the motifs were found with the labels of the fitting rows among them, so the arm's record states the caveat ([the reliability panel](reliability.md)).

The settings of a discovery, such as the widths of a motif and the number kept, are in [`_motifs.yaml`](../../experiments/configs/_motifs.yaml), whose comments give the reason for each.

## Motifs in a run

Both sources are off unless a dataset's [config](../glossary.md#config) extends `_motifs.yaml` beside the shared config of its kind of target. [Run configs](../../experiments/configs/README.md#motifs) says how a config takes the motif sources, including the two settings that the dataset's config must state because they are facts about the dataset: which collection applies, and which strands are scanned. The comments of `_motifs.yaml` give the reason for each of its other settings and for the arms it fits, and [`arms.py`](../../seqmaestro/arms.py) declares which models can take each source, with the reason for each model that cannot.

**The step before the arms.** Before the arms on a motif pool start, the [driver](../glossary.md#driver) runs a step ([`run_motifs.py`](../../experiments/runners/run_motifs.py)) that scans every training and validation sequence once with the known motifs, and makes each discovery that can be made in advance. Each arm on a motif pool waits for the part of the step it needs. A featuriser makes whatever the store lacks, so the step saves work without deciding any value: an arm started without it gets the same features. The [held-out split](../glossary.md#held-out-split) is scanned at the [reveal](../glossary.md#reveal), by the featuriser that each cell saved.

**The environment.** STREME and Tomtom run in an environment of their own, `SeqMaestro_Method_meme`, which `./install.sh --with-motifs` builds ([environments](../../ENVIRONMENTS.md)). The motifs of a discovery come back as text in the MEME format and are scanned in Core (`SeqMaestro_Core`), so a fitted featuriser does not need STREME's environment, and the reduction, the panel and the reveal run without it.
