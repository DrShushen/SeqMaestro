<!-- component: why a black-box model is present and how it is used. Dataset independent.
     Include in Stage 2, with the key table. -->

**XGBoost is included as a performance reference only.** It is a powerful, general-purpose model
(an ensemble of many decision trees) whose internal reasoning cannot be read off. It shows how much
signal a flexible model can extract from each feature source. If an interpretable model scores close
to it, little was lost by insisting on interpretability. If XGBoost scores far above every
interpretable model, the signal has a structure the simpler models cannot express. No feature in
this report is taken from XGBoost.
