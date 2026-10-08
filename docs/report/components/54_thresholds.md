<!-- component: the threshold sweep. Dataset independent. Include in Stage 3B, with the sweep figure. -->

### Why every cut-off is shown

Calling a feature "reproducible" requires a cut-off on how often it must be chosen, and there is no
principled value for it. Moving the cut-off from 0.6 to 0.7 moves features across the line, and
quoting a single cut-off would suggest that line means more than it does.

So the counts are shown at every cut-off, from just over half of the half-samples up to all of them.
The figure has one row per configuration and one column per cut-off, and each cell holds:

| in the cell | what it counts |
|---|---|
| **the large number** | how many of the {{param:n_features}} features the configuration reported clear this cut-off |
| **the small number** | how many of all its candidate features clear it |
| **a ring** | the small number is larger than chance alone would produce at this cut-off (the bound from tier 1) |

**Read each row from left to right.** The counts fall as the cut-off rises, and how quickly they fall
is the result. A configuration that keeps six features at 0.9 is in a much stronger position than one
that keeps six at 0.55 and none at 0.7.

**The small number shows whether anything stable was available.** If no candidate at all clears a
cut-off, the pool contained no stable feature, and the instability comes from the data rather than
from how features were selected.

**A cell without a ring is not a warning about its features.** Rings are missing whenever the
candidate pool is only a few times larger than the number selected, because the chance bound is then
too loose to be informative.

**What to take from the figure.** A feature that survives the whole range of cut-offs is worth more
than one that survives only the loosest. If the counts drop sharply between 0.6 and 0.8, the
selection is close to a coin toss among near-equivalent features, and the property they share is
what to carry forward.
