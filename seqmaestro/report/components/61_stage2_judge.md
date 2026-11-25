<!-- component: the judge's column of the Stage 2 reproducibility summary. Dataset independent.
     Include after 60_stage2_reproducibility.md, only where the run's config ran the judge
     (facts.json: stage2.judged); without it the summary table has no such column. -->

The summary table has one more column, **how meaningful the features look**. A language model rates
each feature from its name and description, without being told where it came from, on a scale from 1
(trivial) to 5 (would change how one thinks about the mechanism), and the column is the average.
The rating is of how interesting a feature would be if it held, whereas Stage 3B tests whether it
holds.
