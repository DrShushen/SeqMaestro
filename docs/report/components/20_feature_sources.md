<!-- component: what a feature and a feature source are. Dataset independent.
     Include in Stage 2, before the key table. -->

### Features, and where they come from

A model cannot read a DNA sequence directly. It reads **features**: numbers computed from each
sequence. "Fraction of C or T in the first 100 bases" is a feature. So is "is the base at position 12
a T?", which takes only the values 0 and 1.

The features a model is given to choose from come from one or more **feature sources**, named
together, as in **positions + motifs**. The study compares two kinds of source.

**Enumerated sources** list every feature of a fixed form, generated mechanically with no reference to
the biology. The **positions** source has one yes/no column for every combination of position and
base in the window. The **motifs** source has a count for each short substring. Such sources have
many columns, each carrying very little on its own, and none was included because anyone expected it
to matter, so they carry no prior knowledge of the biology.

**Written sources** contain features written by a language model. The language model was given a
description of the task and a sample of training sequences, each shown with its target and drawn
evenly across the target's values, and asked to propose measurements that might predict the target. Each proposal is a short piece of code with a
name and a description, so anyone can read what it computes. These sources have far fewer
features, and each one is a stated hypothesis. There are four variants:

- **written**: a single round of proposals.
- **written, iterative**: several further rounds, in which the language model is shown how the
  previous round's features performed in a fitted model.
- **written per split**: a decision tree that asks the language model for new candidate features at
  each branch, based only on the sequences that reach that branch.
- **written terms**: a linear model whose terms the language model writes during the fit, each kept
  only if removing it worsens the predictions on sequences held out from the fit.

The last two are written inside one particular model's fit, so they are only ever given to that
model. The first two are written for each model separately: two models compared on one written
source were given different proposals, and a difference between their results includes the
difference between the proposals.

A written source can also be handed an enumerated one beside it, as in **motifs + written,
iterative**. The model then chooses among both kinds of feature, and in the iterative variant the
language model is shown which of all of them the fitted model leaned on, the enumerated ones
included.

The two kinds differ in what they offer. A written feature is much easier to interpret and act on,
because it comes with a name and an explanation. But when the task description names a protein or a
mechanism, a written feature might restate something the language model already knew from the
literature, where an enumerated feature cannot. Stage 3B tests every feature on held-out sequences,
and that test gives the same answer whether a feature was recalled or newly discovered.
