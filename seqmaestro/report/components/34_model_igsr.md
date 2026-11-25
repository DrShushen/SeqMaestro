<!-- component: the IGSR model and its figure. Dataset independent, for a continuous target, the
     only kind IGSR fits. Include in Stage 3A, immediately before the IGSR weights figure; the
     two directions are explained by components/32_model_l1.continuous.md, which the same figure
     shape follows. -->

### The IGSR model and its figure

**IGSR** (influence-guided symbolic regression) is a linear regression whose features, called
**terms**, a language model writes as short functions of the whole sequence (and of the
measurements supplied beside it, where the configuration includes them). A term can count a motif
anywhere in a sequence of any length, measure a composition over a stretch of it, or combine a
sequence property with a measurement. The language model proposes terms in rounds. After each
round, every term is fitted together with the others, and a term is kept only if the error on the
validation sequences rises when its weight is set to zero (its **influence**). A tree search tries
several such series of proposals and keeps the model with the lowest validation error. The tables
call this source **written terms**.

The model shown here is that configuration reduced to {{param:n_features}} terms, and refitted on
the training sequences with the fit its search used: least squares, with a penalty that shrinks the
weights (ridge or lasso) where the run chose one. Each weight is shown **in the target's units per
standard deviation of its term** across the training sequences, as in the sparse linear model's
figure. The terms are the language model's, and the table after the figure gives each term's
description, as that model wrote it.

**The figure** has the same two panels as the sparse linear model's: on the left each term's
weight, and on the right what the term does on its own, Spearman's rank correlation between the
term and the target across the training sequences. Rows where the model uses a term against the
direction it has by itself are shaded, and, as in that figure, such a term corrects for another
term in the model.
