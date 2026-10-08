<!-- component: the cross-configuration concept map. Dataset independent, for a continuous target;
     its twin for a binary one is 55_concepts.binary.md. Include in Stage 3B, with the concept
     figure. -->

### Ideas, not individual features

Different configurations often find the same property under different names. One might report the
fraction of one class of bases over a stretch of the window, another a count of a few short motifs
over nearly the same stretch, and a third a single base inside it. Counted by name, these look like
three unrelated findings. Asking several independent models the same question is meant to show where
they agree, so the features reported by all configurations are grouped by **what they compute**.
Every feature is evaluated on the same sequences, and two features are grouped when their values track each other
closely, at a rank correlation of {{param:concept_similarity}} or more. Groups are then chained: if A
matches B and B matches C, all three form one group. Each group is called a **concept**, and is named
after one of its members.

Three things to keep in mind:

- **Direction is ignored when grouping.** A measurement and its reverse (for example, the fraction of
  A and the fraction of non-A) carry the same information, so they are grouped together. If a
  concept's members disagree about whether higher values go with a higher or a lower target, the
  concept is shown with no direction.
- **Chaining can merge too much.** A run of borderline matches can join two ideas that are not really
  the same. If a concept's members look like different things, they may well be.
- **Grouping uses the training and validation sequences.** That is the right place to ask whether two
  features compute the same thing. Whether either one holds is a separate question, answered by
  tier 2 on the held-out sequences, and the concept map carries tier 2's results over unchanged.

### The map

Each row is a concept, and each column is a configuration. A dot means that configuration included a
feature from this concept among its final {{param:n_features}}, whatever it called it. The dot's
colour shows whether higher values go with a higher target (blue) or a lower one (red), when the
members agree on this.

The columns on the right summarise the evidence, taking the best value among the concept's members:
the highest selection frequency (tier 1), the strongest rank correlation with the target on
held-out sequences and its corrected p-value (tier 2), and the number of configurations that reached
the concept. In the correlation column, a negative value is a member that points the other way, so
the shading reflects the distance from 0 rather than the number itself.

**Take the correlation together with the p-value.** The p-value says whether a correlation is larger
than chance, and on a large held-out set almost any real correlation passes. The correlation says how
large the effect is, which decides whether an idea is worth an experiment.

**What to look for.**

- A concept reached by several configurations, from more than one model class, that also holds on
  held-out sequences is the strongest kind of result this analysis produces.
- A concept reached by several configurations that does **not** hold on held-out sequences usually
  reflects a shared starting assumption rather than a shared discovery. The configurations agreed
  because they approached the data the same way, and the data did not confirm it.
- A concept reached only once but with strong held-out support is a real association that only one
  configuration happened to express.
