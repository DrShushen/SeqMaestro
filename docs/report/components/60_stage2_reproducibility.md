<!-- component: the condensed Stage 2 reproducibility summary. Dataset independent.
     Include in Stage 2, with the summary table; the full tables go in an appendix. -->

### Does a configuration find the same features each time?

Each configuration was run {{param:n_repeats}} times, and the table below summarises how much those
runs agreed about which features matter. It is kept short because Stage 3B answers the same question
much more thoroughly, with {{param:half_samples}} reruns instead of {{param:n_repeats}}. The full
tables are in Appendix A.

| column | what it measures | how to read it |
|---|---|---|
| **same features again** | how often two repeats pick the same top features, corrected for the agreement the same configuration reaches on shuffled labels | 0 means no more agreement than on shuffled labels, 1 means every repeat picked the same set, and below 0 means less agreement than on shuffled labels |
| **ideas found in more than one repeat** | for written sources only: features are grouped by what they compute, and this counts how many of those groups appear in more than one repeat, out of all the groups found | a high count beside a low "same features again" means the idea recurs while its exact wording changes |
| **found by another feature source** | the share of this configuration's features that a configuration using a different feature source also found | agreement across feature sources carries more weight, because configurations drawing on the same columns can agree simply by sharing them |

Two cautions apply. Agreement across only {{param:n_repeats}} repeats is a rough estimate. And a
language model tends to propose similar features whatever the labels say, so written sources agree
with themselves even when there is no signal. That is why agreement is measured against what the
same configuration reaches on shuffled labels.
