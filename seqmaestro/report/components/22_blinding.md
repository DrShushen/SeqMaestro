<!-- component: blinding, what a blinded run kept from the language models. Dataset independent.
     Include in Stage 1, under "What the models were told", straight after naming the withheld
     fact, when facts.json's stage1.blinding.blinded is true. -->

### What the language models were not told

This run was **blinded**. The language models that wrote features were given a description of the
task with one fact left out, the fact named just above. Before the run started, the column names
the models were shown, and the descriptions of the measurements, were checked for anything that
would give the fact away. Where the run asked a language model to rate how meaningful each feature
would be, that model was told the fact, since it cannot say what a feature would mean without
knowing what is being predicted.

A fact is withheld when a model could look up an answer from it instead of finding one in the data:
the name of the protein whose loss defines the target, for example, or the two cell types being
compared. A model told the cell types can write features for the factors it knows act in them,
whatever the sequences hold. In a blinded run, a feature that names such a factor had to be reached
from the sequences, their targets and the rest of the description.

**Limitations.** A model can still recognise a known motif in the sequences it is shown, and name
the factor that binds it. Blinding therefore narrows what a model can bring in from outside, and a
feature that names a known factor may still owe something to what the model already knew.
