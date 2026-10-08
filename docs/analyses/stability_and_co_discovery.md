# Stability and co-discovery

> **Maintained page.** Change it in the same commit as `stability.py` or `co_discovery.py` under
> `seqmaestro/pipeline/analyses/`, which the scripts `analyse_stability.py` and
> `analyse_co_discovery.py` call, or `seqmaestro/harness/stability.py`.

Two analyses, each answering one question, share most of their machinery.

- **Stability** (`analyse_stability.py`): when the *seed* changes, does one arm report the same
  features again?
- **Co-discovery** (`analyse_co_discovery.py`): when the *arm* changes, do several of them land on
  the same property?

The shared machinery (evaluating every feature on common rows, matching two sets, chaining matches
into clusters, the top-N cut, the permuted null) is in [`README.md`](README.md). Read that first.
This page covers what each analysis does *with* it.

Worked numbers below are illustrative, taken from one finished run of a nine-arm grid. They are
examples of the arithmetic and make no claim about any dataset.

---

## Stability, step by step

Input: **one arm, its seeds**, one label condition. Stability is a property of a set of runs, so an
arm with fewer than two seeds is skipped, and a run in which every arm is skipped writes no table;
the report then says stability was not measured.

1. **Restrict each seed to its top 20** by that arm's own importance.
2. **Pairwise Jaccard.** For each seed pair, match the two sets and compute
   `matched / (|A| + |B| − matched)`. Average over the pairs. This is `top_n_jaccard`. Two seeds
   that both selected nothing have no feature to agree on, and their pair scores 0. An arm whose
   shuffled-label fits select nothing therefore has a chance level of 0, and its kappa is its
   real-label agreement.
3. **Also, reported only as a diagnostic:** cluster across all seeds to get a runs-by-clusters
   binary matrix, and compute the **Nogueira estimator** `phi` on it. Phi is the established
   measure, but it needs a universe of known size `d`, which a generative arm does not have. Taking
   the union of discovered features as `d` is a convention, and phi values computed under two
   different choices of `d` are not comparable. `full_phi` and `top_n_phi` are written, never
   quoted.
4. **The chance level.** Repeat steps 1 and 2 on the same arm's **permuted-label** seeds.
5. **Correct.** `top_n_kappa = (J_real − J_permuted) / (1 − J_permuted)`.

**`top_n_kappa` is the column to report.** It is the only one that means the same thing for every
arm: 0 when an arm reproduces its features no better than on shuffled labels, 1 when the seeds
returned identical sets. It is carried on both rows of each arm, so a reader who filters to one
label condition still sees it.

| arm | J real | J permuted | kappa |
|---|---:|---:|---:|
| `ebm/positional` | 0.3881 | 0.0366 | (0.3881 − 0.0366) / (1 − 0.0366) = **0.3649** |
| `deft_or_cart/positional` | 0.1163 | 0.0181 | (0.1163 − 0.0181) / (1 − 0.0181) = **0.1000** |

**Kappa is computed for the top-N view only.** `ebm/positional`'s *full* Jaccard is 1.0000 under
both conditions, because the EBM keeps every feature it is given and so agrees with itself perfectly
on noise too. Dividing by `1 − chance` would then explode or be undefined. That is an artefact of
comparing sets of very different sizes, which the top-N view removes.

---

## Co-discovery, step by step

Input: **every arm within one (label condition, seed)**.

1. **Restrict each arm to its top 20.**
2. **Cluster all of them together.** Nine arms × 20 features = 180 features into one graph. A
   cluster is now called a **concept**: a property, held by however many arms found it.
3. **Score each concept** by who is in it:
   - `n_arms`: how many arms, raw.
   - `n_witnesses`: the same after collapsing each group of **definitionally linked** arms to one.
     The tree on `positional` (CART) and on `positional+llm_generated_joint` (DEFT) share the tree,
     the naive enumeration and, under `grid.permutation_scope: seed`, the shuffled labels, so they
     agree because of the pipeline, and
     their agreement says nothing about the data. The same holds for the tree on any other pool
     with and without that source, such as the pair with the covariates (`linked_by_fit`).
     Arms that consumed one cached proposal would be linked in the same way, but each arm's
     proposal is made for that arm, so no two arms of a run share one.
   - `n_pools`: how many **distinct input pools** are represented. This is the stricter count,
     because three models over one vocabulary are three arms but one representation.
   - `direction`, `direction_agreement`: whether the arms that found it also agree about which
     class it argues for. See below.
4. **Pool over seeds.** Each arm appears in 20 concepts per seed.
5. **Per arm, count how often it had company**, at rising strengths: the share of its concepts with
   `n_witnesses` ≥ 2, ≥ 3, ≥ 4, and the share where a *different pool* also found it.
6. **The chance level and the correction**, as in stability.

**Read `cross_pool_share` when comparing arms with each other.** Corroboration by any second arm is
a weak criterion, because it fires on some concepts even under shuffled labels. It also
flatters whichever arm has the most company, since three models over one vocabulary corroborate each
other by construction. The cross-pool column counts only arms handed a *different* input pool.

**Worked example**, one seed of real labels, in which 180 features collapse to 121 concepts:

| arms in the concept | concepts |
|---|---:|
| 1 (nobody else found it) | 84 |
| 2 | 24 |
| 3 | 9 |
| 4 | 3 |
| 8 | 1 |

The single 8-arm concept held `pos_50_A`, `pos_50_is_A`, `central_base_is_a` and
`a_run_length_crossing_center`: one property, reached independently by a fixed vocabulary, a
vendored tree and a language-model proposer.

### Direction agreement, and the sign trap

Finding the same concept is weaker than agreeing about **which way it runs**. Two arms can both lean
on the GC content of one window and use it in opposite directions, which corroborates nothing. So
each concept also carries the direction its members agree on and the share holding it.

**Signs are aligned before they are compared, and skipping that would invert the answer.** Features
are matched on *absolute* rank correlation, so a cluster may legitimately hold a GC fraction from
one arm and an AT fraction from another, whose directions are opposite while the finding is the
same. Each member is therefore read against the first one and flipped where it runs the other way.

What is compared is the **model's** `direction`, not `marginal_effect`. Two features that matched
already have near-identical marginal effects by construction, so agreeing about those would only
restate the match. Agreeing about how each arm *uses* the property is a real claim, because every
arm conditions on a different set of other features.

`direction_agreement` is blank for a concept that only one arm gave a direction for, because there
is nothing to agree with, and 1.0 would read as unanimity.

> **When this column is informative.** A model can only differ from the plain data association if it
> has something to trade against. On a pool of nearly independent features, such as positional
> indicators, every arm's conditional effect collapses onto the marginal one and they all agree by
> default, so 1.0 is the *expected* value and not a result. The column is informative on cross-pool
> concepts and inside pools of correlated features, where disagreement is possible.

---

## Where they differ

| | stability | co-discovery |
|---|---|---|
| what varies | the **seed** | the **arm** |
| what is held fixed | the arm | the seed |
| how sets are compared | mean pairwise Jaccard over seed pairs | one clustering, count arms per concept |
| clustering used for | `phi` and diagnostics only | the headline; it *is* the unit of analysis |
| unit reported | one number per arm | one row per arm, plus one row per concept |
| the null | same arm, shuffled labels | same arms, shuffled labels |

Stability quotes the pairwise Jaccard and uses clustering only for the diagnostic phi. Co-discovery
rests entirely on the clustering, so its transitivity risk is the one to watch.

---

## Where the null breaks

Both corrections assume the permuted run differs from the real one **only in the labels**. That
holds for an arm selecting from a fixed vocabulary: shuffle the labels and it still selects 20
features from the same pool.

It does not hold for a **generative** arm, whose feature *mix* itself depends on the labels. DEFT's
arm (`deft_or_cart` on a pool with `llm_generated_joint`) uses far fewer language-model features
under shuffled labels than under real ones, because those candidates only win splits when there is
signal to win with. Its null is therefore very nearly the CART arm and not the same arm, and
subtracting it compares two different mixtures. Holding the mixture fixed can flip the sign of its
corrected figure, which is the signature of a composition effect and not of a finding.

Two consequences follow:

- **Split by provenance before reading a mixed arm's row.** Selected and generated features have
  different corroboration rates and different nulls.
- **The correction divides by `1 − chance`.** An arm whose chance level is high turns a small
  shortfall into a large negative. For example, a gap of −0.39 against a chance level of 0.80 gives
  −0.39 / 0.20 = −1.95. That is why the per-arm row reports observed and chance side by side and
  puts the corrected scalar last.

---

## Caveats that apply to both

**Everything rests on the top 20**, which comes from each arm's own importance ranking, and those
rankings are not comparable between models. An arm whose importance ranks the wrong features will
look unstable and uncorroborated whatever it actually found.

**The reference split must be identical across every arm compared.** Otherwise matching is
meaningless, and the code refuses to compare runs whose reference rows differ in number.

**Co-discovery's per-arm number depends on every other arm.** Concepts are connected components over
the pooled feature sets, so adding or improving one arm changes every arm's row.

**`n_pools` counts the pools the runs record**, as their members. A directory's name is only a
label, and counting labels could make one representation look like two. A cell whose run states no
pool is refused, and every cell's `run.json` records one, as `sources`.

**Where the tables are.** Stability writes `results/stability.csv`, one row per arm. Co-discovery
writes `results/co_discovery.csv`, one row per concept, and `results/co_discovery_by_arm.csv`.
