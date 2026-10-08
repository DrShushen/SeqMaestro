# The held-out split, and the one reveal

> Tracks `seqmaestro/harness/heldout.py`, `seqmaestro/pipeline/reveal.py`, the sides that predict
> a cell (`seqmaestro/pipeline/sides.py`, with `predict_cell` in each module of
> `seqmaestro/pipeline/vendored/`), the validation-only scoring in `seqmaestro/harness/runner.py`
> and `seqmaestro/harness/bundle.py`, the predict modes of `experiments/adapters/deft_bundle.py`
> and `deft_cart.py`, the `predict` and `evaluate` modes of `experiments/adapters/igsr_bundle.py`,
> and `reveal_tier2` in `seqmaestro/reliability/panel.py`. Change one, change this page.

A run makes many choices: which features an arm keeps, what the analyses cluster, and, later, which
N features survive the reduction. Every one of them is made on the training and validation rows. The test split is read **once**, at the end, when every model the run
saved predicts it and the predictions are scored. Nothing before that point has seen a test number,
so nothing before it can have been steered by one.

Scoring the test split inline and hiding the numbers from the later stages would keep them out of
sight only by discipline. Here they do not exist until the reveal.

## 1. The rule

**Only `seqmaestro.harness.heldout` reads `test.csv`**, and only the reveal
(`seqmaestro/pipeline/reveal.py`, which `run_reveal.py` starts) calls it. Stage 1
creates the split and is exempt. `tests/test_heldout.py` enforces the rest statically: a module in
`seqmaestro/` or `experiments/{runners,analysis,adapters}/` that names the test split fails the
suite. Code that reads test results after the reveal takes the names from `heldout` (`TEST_SPLIT`,
`TEST_ARRAYS`) and checks `reveal_record` first, so every dependency on the split shows up as an
import of that one module.

**This is hygiene, not a security boundary.** `test.csv` sits in `data/` and anyone can open it.
What the arrangement prevents is reading it by accident, and a test number deciding something the
run was meant to decide without one.

## 2. A cell before the reveal

| file | holds |
|---|---|
| `metrics.json` | the `val` block, scored by the dataset's kind of target; for a binary target also `threshold_chosen_on`, and the block's `threshold` is the cut chosen on validation; for a model that can be undefined on a row, `filled_rows`: how many validation rows took its constant prediction |
| `predictions.npz` | `val_true`, `val_score` |
| `run.json` | what was run, including `seed`, `n_bootstrap` and `target_kind`, which the reveal scores the test block with |
| `fitted.pkl` | the featuriser and model, which the reveal predicts with. A DEFT or an IGSR cell has none: its `run.json` names its model in `fitted_model`, a file in its bundle for a Stage 2 cell (DEFT's tree, IGSR's `model.json`) and in the cell for one fitted later (`model.dill`, `model.pkl`) |

The fitted model is mandatory. A cell that saved none could never be scored on the test split, so
`run_model` saves it before writing anything else and `read_bundle` refuses a bundle without one.

## 3. The reveal

`experiments/runners/run_reveal.py --run-dir <run>`. The driver runs it as a stage, once every stage
before it has finished clean (`--stop-before-reveal` holds it back, so a run can be checked first),
and a revealed run resumes from it. `--check` makes the first step alone: it says how many items
are ready, lists the rest, seals and opens nothing, and exits 2 when something is not ready. The
reveal:

1. **Checks the run is finished.** Every cell its recorded `config.yaml` should have produced must
   exist, carry a validation block, record `seed`, `n_bootstrap` and the dataset's `target_kind`,
   and hold the saved model its `predict_with` names. Those cells are one baseline cell per seed, every arm's cells for every label and seed, and
   each reduced arm's three models ([`reduction.md`](reduction.md)). Every arm the reliability panel
   profiles must have its record, which lists that arm's leave-one-out refits, each a cell held to
   the same standard ([`reliability.md`](reliability.md)). Otherwise it refuses and lists what is
   missing. `--override-incomplete REASON` reveals what is ready and records the reason and what
   was missing, because a reveal over a partial run is a fact about every number it produces.
2. **Writes `reveal.json`** before any test row is read. It records the time, the commit and whether
   the tree was dirty, the test file's recorded digest, the list of items sealed (the cells, and
   each profiled arm's panel), anything found on disk that the config does not account for
   (recorded, and left unscored), and the override if there was one.
3. **Opens the test split**, checking the file against the digest `dataset.json` recorded for it.
   Nothing earlier checks that digest, because nothing earlier opens the file.
4. **Predicts and scores each sealed cell.** A cell with `fitted.pkl` is loaded and predicts in the
   Core environment. A DEFT cell is predicted on DEFT's side of the boundary, by the adapter its
   `run.json` names in `predict_with`. The test sequences go across numbered and without labels,
   and the adapter loads the saved tree. For a reduced tree it first evaluates the arm's features
   with the Stage 2 tree that owns them. An IGSR cell is predicted the same way by IGSR's adapter,
   from the model its bundle saved. A reduced or refitted IGSR cell has its terms evaluated on
   IGSR's side, and its least-squares model, saved on this side, predicts from them. The scores
   come back beside the row numbers, which are checked before anything is scored. Each test block
   is computed by the metrics of the dataset's kind of target (`seqmaestro.target`), with a binary
   target's validation threshold and the cell's seed and bootstrap count, and written into the
   cell's files: `metrics.json` gains `test`, and `predictions.npz` gains `test_true` and
   `test_score`. For a model that can be undefined on a row, such as a PySR equation that divides
   by a feature, `filled_rows` gains how many test rows took the model's constant prediction.
   The count is made from the same transform of the test rows as the scores.
5. **Computes each profiled arm's Tier 2 on the test split**, the one piece of test-side work that
   is not a model. The arm's candidates, after the panel's prefilter, are evaluated on the test
   rows by what they came from: the featuriser, the DEFT tree, or IGSR's adapter (a method's side
   again gets the rows numbered and unlabelled). Each feature's association is calibrated under
   the real labels, a permutation of them, and, on the check arm, the planted signal. The results
   go into the arm's panel directory. On a dataset fitted with per-row weights this step computes
   nothing, because the panel has no Tier 2 for a weighted design ([`reliability.md`](reliability.md)).
6. **Marks the reveal complete** in `reveal.json`, with how many items of each kind it did.

**Nothing is refitted.** **It is resumable:** run again, it does what is sealed and not yet done
and nothing else, because the sealed list was fixed in step 2. After it, the panel's Tier 3 and its
tables are assembled from what it wrote, by `run_reliability.py --after-reveal`.

## 4. Why the numbers are the ones inline scoring would have produced

The test block needs four things: the test targets, the model's scores on the test rows, a binary
target's threshold, and the bootstrap draws. The labels come from the same file. The scores come from the
same fitted objects, restored from `fitted.pkl` or the saved tree. The threshold was chosen on
validation, is stored in the validation block, and survives the JSON round trip exactly. The
bootstrap is seeded by the cell's own seed and runs the cell's own count, through the same function.
So the block is the same, and so are the files: a revealed cell is byte for byte the cell that
scoring the test split inline would have written. `tests/test_heldout.py` checks this for each
Core model of a binary target.

## 5. There is no going back

Once `reveal.json` exists, every stage that fits or selects refuses to run in that directory
(`refuse_if_revealed`), and so does the driver's preflight. Adding an arm to a revealed run means a
new run. `tests/test_heldout.py` checks that the module behind every `run_*.py` script but the
reveal's calls `refuse_if_revealed`. The reliability panel does so before the reveal. Its
`--after-reveal` mode fits and selects nothing, reads only what the reveal wrote, and refuses the
opposite case: a run that has not been revealed.

## 6. Reading test numbers afterwards

After the reveal, a cell's test block is in its `metrics.json` under `TEST_SPLIT`, and
`revealed_metrics` returns it, refusing before the reveal. `export_features_xlsx.py` reports the
headline of the run's kind of target (AUROC or R²) on test for a revealed run and on validation
otherwise, with the split named in every header.

A run finished before the reveal existed carries its test blocks inline and has no `reveal.json`.
Such runs are not migrated.
