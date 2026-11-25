<!-- component: the motif sources. Dataset independent.
     Include in Stage 2, after components/20_feature_sources.md and any component on covariates,
     when a configuration's feature sources include motifs. -->

### Motifs

A **motif** is a pattern of bases a protein binds, written as a table: for each position of a
binding site, how often each base occurs there. A motif scores every stretch of a sequence as long
as itself, and a stretch scores high where its bases are those the motif expects. Each score is
turned into a p-value, the chance that a stretch of random sequence scores at least as high: a match
at p = 0.0001 is one that random sequence reaches once in ten thousand stretches. On DNA, each
stretch is scored on both strands, since a protein can bind either. On RNA, which has one strand,
it is scored as given.

Each motif gives two features:

- **best match**: how strong the motif's best match anywhere in the sequence is, as -log10 of its
  p-value. A value of 4 means the best match has p = 0.0001.
- **matches**: how many places in the sequence match the motif at the threshold the run sets, which
  the feature's description states.

The **known motifs** source holds every motif of a published collection, such as JASPAR's for the
transcription factors of a group of species, or CisBP-RNA's for the RNA-binding proteins of one
species. Each motif was measured for one protein, in an experiment, and is named after it. This
source therefore brings in what is already known. A model that relies on a known motif indicates
that the target goes with that protein's binding preference, and a reader can recall what is known
about the protein, as a language model can. Proteins of one family often bind alike, so a motif
named after one protein stands for its family, and which member binds the sequence is a further
question.

The **discovered motifs** source holds the motifs a discovery tool, STREME, found in the training
sequences a model was fitted on: patterns more common in one class than in the other, or at one end
of the target than at the other, in either direction. Each is named for how it was found, as in
**streme up 1 GATTACA**: enriched in the positive class or at the high end of the target ("up"),
its rank among the motifs found that way, and its consensus, the commonest base at each position,
with the codes of the IUPAC alphabet where two or more bases are common (N for any). Such a motif
is not named after a protein. Where the run compared the motifs found with a published collection,
the table of a model's features names the known motif closest to a discovered one, provided the two
are alike: the comparison, made with Tomtom, must give the pair a q-value of 0.05 or below. A
discovered motif with no such neighbour does not closely resemble any motif of the collection. A name given
this way is a lead for a reader to follow up, and it was not shown to any model. The motifs were
found using the targets of the training sequences, so a discovered motif is tested, as every
feature is, on the held-out sequences, which the discovery never saw.
