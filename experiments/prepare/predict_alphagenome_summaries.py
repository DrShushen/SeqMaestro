#!/usr/bin/env python
"""AlphaGenome's predictions for a set of genomic elements, each track summarised over the element.

Runs in `SeqMaestro_Method_alphagenome`, which holds AlphaGenome's client, and calls its API once per
element. The API key is read from the environment variable `--key-variable` names:

    eval "$(./install.sh --interpreters)"    # AG, where the environment is built
    set -a && source .env && set +a
    $AG experiments/prepare/predict_alphagenome_summaries.py \
        --elements data/<dataset>/elements.csv --id-column <id> --sequence-column sequence \
        --chrom-column chrom --start-column start --end-column end --organism mouse \
        --outputs DNASE ATAC CHIP_TF CHIP_HISTONE CAGE --context native --model held_out \
        --folds data/<dataset>/alphagenome/sequences_mouse.bed.gz \
        --key-variable AG_KEY --out-dir data/<dataset>/alphagenome/native_held_out

The start and end columns must be one-based with an inclusive end, so that they span as many bases
as the element's sequence holds, and an element whose coordinates span another number is refused. A
BED file's coordinates are zero-based with an exclusive end, and need one added to each start.

**What AlphaGenome is given (`--context`).** Its shortest input is `WINDOW` bp, so each element is
placed inside a longer input. `native` gives it the `WINDOW` bp of the reference genome centred on
the element, which includes the element's genomic neighbourhood. `scaffold` gives it the element's
sequence centred in flanks cut from shuffles of the element that keep the element's dinucleotide
counts, so the only genomic sequence the model sees is the element.

**Which model predicts (`--model`).** `all_folds` is AlphaGenome's default model, distilled from
models trained on the whole genome, the elements' loci included. `held_out` predicts each element
with the fold model whose training excluded the region the element's native window lies in. The
regions and their folds come from AlphaGenome's training setup (`--folds`, the BED file that
`alphagenome.data.fold_intervals` loads), and the client states which folds each model held out.
An element whose window overlaps regions of folds held out by different models is refused, because
every model trained on part of that window. An element whose window overlaps no region was in no
model's training data, and is predicted by the first fold model. Both contexts choose the model
from the native window, so an element gets the same model in either context.

**Each track is summarised over the element at that track's resolution**: its mean and its maximum
over the bins that cover any part of the element. A stranded track has one column per strand.

Writes, in `--out-dir`:

    summaries.tsv.gz   one row per element, one column per output type, track, strand and statistic
    columns.tsv        what each column holds: output type, track, strand, statistic
    tracks.tsv         AlphaGenome's metadata for every track of the requested outputs
    elements.tsv       per element: the model that predicted it, its folds, where it sat in the input
    provenance.json    the settings, the client's version, the input digests, the time

Each prediction is summarised and appended to `partial.jsonl` as it arrives, so an interrupted run
resumes where it stopped. The tables are written only once every element has been predicted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

#: AlphaGenome's shortest input. Every element must fit inside it.
WINDOW = 16_384

ALPHABET = frozenset("ACGT")
CONTEXTS = ("native", "scaffold")
MODELS = ("all_folds", "held_out")
ORGANISMS = {"mouse": "MUS_MUSCULUS", "human": "HOMO_SAPIENS"}
PARTIAL, SETTINGS = "partial.jsonl", "settings.json"


def dinucleotide_shuffle(sequence: str, rng: np.random.Generator) -> str:
    """A random sequence with the same dinucleotide counts, first letter and last letter as
    `sequence` (Altschul and Erickson, 1985).

    The letters are the nodes of a graph, and each dinucleotide is an edge from its first letter to
    its second, so the sequence is a walk that uses every edge once. The shuffle draws another such
    walk uniformly at random. For every letter except the last, it chooses the edge by which the walk
    leaves that letter for the final time, and keeps a set of choices only when those edges lead to
    the last letter without a cycle, which guarantees the walk can use every edge. The remaining
    edges of each letter are then taken in random order.
    """
    if len(sequence) < 3:
        return sequence
    edges: dict[str, list[str]] = {}
    for first, second in pairwise(sequence):
        edges.setdefault(first, []).append(second)
    final = sequence[-1]
    # Drawn again until the last exits reach the final letter from every letter without a cycle.
    while True:
        last_exit = {letter: int(rng.integers(len(out))) for letter, out in edges.items()
                     if letter != final}
        if all(_reaches(letter, final, edges, last_exit) for letter in last_exit):
            break
    order: dict[str, list[str]] = {}
    for letter, out in edges.items():
        rest = list(out)
        chosen = rest.pop(last_exit[letter]) if letter in last_exit else None
        rng.shuffle(rest)
        order[letter] = rest + ([chosen] if chosen is not None else [])
    walked, position = [sequence[0]], dict.fromkeys(order, 0)
    for _ in range(len(sequence) - 1):
        here = walked[-1]
        walked.append(order[here][position[here]])
        position[here] += 1
    return "".join(walked)


def _reaches(letter: str, final: str, edges: dict[str, list[str]], last_exit: dict[str, int]) -> bool:
    """Whether following last exits from `letter` arrives at `final` without revisiting a letter."""
    seen = set()
    while letter != final:
        if letter in seen:
            return False
        seen.add(letter)
        letter = edges[letter][last_exit[letter]]
    return True


def scaffold(sequence: str, rng: np.random.Generator) -> tuple[str, int]:
    """The element centred in `WINDOW` bp, its flanks cut from dinucleotide shuffles of the element
    laid end to end. Returns the input and the position the element starts at."""
    extra = WINDOW - len(sequence)
    left = extra // 2

    def flank(length: int) -> str:
        pieces, total = [], 0
        while total < length:
            pieces.append(dinucleotide_shuffle(sequence, rng))
            total += len(sequence)
        return "".join(pieces)[:length]

    return flank(left) + sequence + flank(extra - left), left


def native_window(start: int, end: int) -> tuple[int, int]:
    """The `WINDOW` bp centred on an element at one-based inclusive `start` and `end`, as a
    zero-based start, and the position the element starts at inside it.

    The window is centred the way AlphaGenome's `Interval.resize` centres an unstranded interval,
    rounding the centre of an odd width up, so it matches the window that call returns.
    """
    first, stop = start - 1, end
    window_start = -(-(first + stop) // 2) - WINDOW // 2
    return window_start, first - window_start


def span_mismatches(elements: pd.DataFrame, columns: dict[str, str]) -> list[str]:
    """The elements whose coordinates, read as one-based with an inclusive end, span a different
    number of bases than their sequence holds. Zero-based coordinates with an exclusive end span one
    base more, and would put the base before each element into every summary of it."""
    spans = elements[columns["end"]] - elements[columns["start"]] + 1
    lengths = elements[columns["sequence"]].str.len()
    return elements.loc[spans != lengths, columns["id"]].tolist()


def element_bins(offset: int, length: int, resolution: int, n_bins: int) -> tuple[int, int]:
    """The bins of a track at `resolution` bp per bin that cover any part of the `length` bp
    starting at `offset`, as a half-open range."""
    low, high = offset // resolution, -(-(offset + length) // resolution)
    if low < 0 or high > n_bins or high <= low:
        raise ValueError(f"The element at {offset}..{offset + length} does not lie inside the "
                         f"{n_bins} bins of {resolution} bp the prediction holds.")
    return low, high


def summarise(output, requested: list[str], offset: int, length: int) -> tuple[dict, dict]:
    """The mean and maximum of every track of every requested output over the element, keyed
    `OUTPUT|track|strand|statistic`, and each output's track list as (name, strand) pairs."""
    values, tracks = {}, {}
    for name in requested:
        data = getattr(output, name.lower())
        if data is None:
            raise ValueError(f"The prediction holds no {name}.")
        array = np.asarray(data.values, dtype=np.float64)
        if array.shape[0] * data.resolution != WINDOW:
            raise ValueError(f"{name} holds {array.shape[0]} bins of {data.resolution} bp, "
                             f"which is not the {WINDOW} bp requested.")
        low, high = element_bins(offset, length, data.resolution, array.shape[0])
        part = array[low:high]
        keys = list(zip(data.metadata["name"].astype(str), data.metadata["strand"].astype(str),
                        strict=True))
        if len(set(keys)) != len(keys):
            raise ValueError(f"{name} lists a track name twice on one strand.")
        for (track, strand), mean, peak in zip(keys, part.mean(axis=0), part.max(axis=0),
                                               strict=True):
            values[f"{name}|{track}|{strand}|mean"] = float(mean)
            values[f"{name}|{track}|{strand}|max"] = float(peak)
        # Lists, because JSON returns tuples from `partial.jsonl` as lists, and a resumed run
        # compares fresh records with loaded ones.
        tracks[name] = [list(key) for key in keys]
    return values, tracks


def fold_models() -> dict[str, str]:
    """Each fold, mapped to the fold model that held it out of training, as the client records it."""
    from alphagenome.data import fold_intervals
    from alphagenome.models import dna_client

    held: dict[str, str] = {}
    for version in dna_client.ModelVersion:
        if version is dna_client.ModelVersion.ALL_FOLDS:
            continue
        for subset in (fold_intervals.Subset.VALID, fold_intervals.Subset.TEST):
            for fold in fold_intervals.get_fold_names(version, subset):
                if fold in held:
                    raise ValueError(f"{fold} is held out of both {held[fold]} and {version.name}.")
                held[fold] = version.name
    return held


def assign_models(elements: pd.DataFrame, chrom: str, folds: pd.DataFrame,
                  held: dict[str, str]) -> tuple[list[str], list[str]]:
    """The fold model for each element, and the folds its native window meets."""
    first_model = sorted(set(held.values()))[0]
    models, met = [], []
    for chromosome, window_start in zip(elements[chrom], elements["window_start"], strict=True):
        near = folds[(folds["chrom"] == chromosome) & (folds["start"] < window_start + WINDOW)
                     & (folds["end"] > window_start)]
        touched = sorted(set(near["fold"]))
        unknown = [fold for fold in touched if fold not in held]
        if unknown:
            raise ValueError(f"No fold model holds out {unknown}.")
        holders = {held[fold] for fold in touched}
        if len(holders) > 1:
            raise ValueError(f"A window on {chromosome} at {window_start} meets folds {touched}, "
                             f"which no single model held out.")
        models.append(holders.pop() if holders else first_model)
        met.append(",".join(touched))
    return models, met


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elements", type=Path, required=True, help="a CSV of the elements")
    for column in ("id", "sequence", "chrom", "start", "end"):
        parser.add_argument(f"--{column}-column", required=True,
                            help=f"the column holding each element's {column}")
    parser.add_argument("--organism", choices=sorted(ORGANISMS), required=True)
    parser.add_argument("--outputs", nargs="+", required=True,
                        help="AlphaGenome output types, e.g. DNASE ATAC CHIP_TF CHIP_HISTONE CAGE")
    parser.add_argument("--context", choices=CONTEXTS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--folds", type=Path,
                        help="AlphaGenome's regions and their folds, as BED; needed by held_out")
    parser.add_argument("--seed", type=int, help="for the scaffold's shuffles; needed by scaffold")
    parser.add_argument("--key-variable", required=True,
                        help="the environment variable that holds the API key")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", help="predict only these elements, for a trial")
    parser.add_argument("--workers", type=int, default=4, help="requests in flight at once")
    args = parser.parse_args()

    if args.model == "held_out" and args.folds is None:
        parser.error("--model held_out needs --folds.")
    if args.context == "scaffold" and args.seed is None:
        parser.error("--context scaffold needs --seed.")
    key = os.environ.get(args.key_variable)
    if not key:
        print(f"The environment variable {args.key_variable} is not set.", file=sys.stderr)
        return 2

    from alphagenome import __version__ as client_version
    from alphagenome.data import genome
    from alphagenome.models import dna_client

    columns = {name: getattr(args, f"{name}_column") for name in ("id", "sequence", "chrom",
                                                                    "start", "end")}
    elements = pd.read_csv(args.elements, dtype={columns["id"]: str, columns["sequence"]: str})
    missing = [column for column in columns.values() if column not in elements.columns]
    if missing:
        print(f"{args.elements} has no column {', '.join(missing)}.", file=sys.stderr)
        return 2
    if args.ids:
        unknown = sorted(set(args.ids) - set(elements[columns["id"]]))
        if unknown:
            print(f"No element is called {unknown}.", file=sys.stderr)
            return 2
        elements = elements[elements[columns["id"]].isin(args.ids)]
    elements = elements.reset_index(drop=True)
    if elements[columns["id"]].duplicated().any():
        print(f"{args.elements} repeats an identifier.", file=sys.stderr)
        return 2
    lengths = elements[columns["sequence"]].str.len()
    if lengths.max() >= WINDOW:
        print(f"An element of {lengths.max()} bp does not fit in {WINDOW} bp.", file=sys.stderr)
        return 2
    bad = [element for element, sequence in zip(elements[columns["id"]], elements[columns["sequence"]], strict=True)
           if set(sequence) - ALPHABET]
    if bad:
        print(f"{bad[:5]} hold letters other than ACGT.", file=sys.stderr)
        return 2
    mismatched = span_mismatches(elements, columns)
    if mismatched:
        print(f"{len(mismatched)} elements, for example {mismatched[:3]}, span a different number of "
              f"bases from {columns['start']} to {columns['end']} than their sequence holds. The "
              f"coordinates must be one-based with an inclusive end. Zero-based coordinates with an "
              f"exclusive end, as in a BED file, need one added to each start.", file=sys.stderr)
        return 2

    placed = [native_window(start, end) for start, end
              in zip(elements[columns["start"]], elements[columns["end"]], strict=True)]
    elements["window_start"] = [start for start, _ in placed]
    elements["genomic_offset"] = [offset for _, offset in placed]
    elements["genomic_length"] = elements[columns["end"]] - elements[columns["start"]] + 1

    requested = list(dict.fromkeys(args.outputs))
    output_types = [dna_client.OutputType[name] for name in requested]
    organism = dna_client.Organism[ORGANISMS[args.organism]]
    if args.model == "held_out":
        folds = pd.read_csv(args.folds, sep="\t", names=["chrom", "start", "end", "fold"])
        elements["model"], elements["folds"] = assign_models(elements, columns["chrom"], folds,
                                                             fold_models())
    else:
        elements["model"], elements["folds"] = "ALL_FOLDS", ""

    settings = {
        "elements": str(args.elements), "elements_sha256": digest(args.elements),
        "folds": None if args.folds is None else str(args.folds),
        "folds_sha256": None if args.folds is None else digest(args.folds),
        "columns": columns, "organism": args.organism, "outputs": requested,
        "context": args.context, "model": args.model, "seed": args.seed, "window": WINDOW,
        "ids": args.ids, "client_version": client_version,
    }
    if (args.out_dir / "summaries.tsv.gz").exists():
        print(f"{args.out_dir} already holds finished predictions; use another --out-dir.",
              file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings_path, partial_path = args.out_dir / SETTINGS, args.out_dir / PARTIAL
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        print(f"{args.out_dir} holds predictions made with other settings; use another --out-dir.",
              file=sys.stderr)
        return 2
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    done: dict[str, dict] = {}
    if partial_path.exists():
        for line in partial_path.read_text().splitlines():
            record = json.loads(line)
            done[record["id"]] = record
    todo = elements[~elements[columns["id"]].isin(done)]
    print(f"{len(elements)} elements, {len(done)} already predicted, {len(todo)} to predict "
          f"({args.context}, {args.model})", flush=True)

    clients: dict[str, object] = {}
    for model in sorted(set(todo["model"])):
        clients[model] = dna_client.create(key, model_version=dna_client.ModelVersion[model])

    def predict(row: pd.Series) -> dict:
        element = row[columns["id"]]
        if args.context == "native":
            interval = genome.Interval(row[columns["chrom"]], int(row["window_start"]),
                                       int(row["window_start"]) + WINDOW)
            output = clients[row["model"]].predict_interval(
                interval=interval, organism=organism, requested_outputs=output_types,
                ontology_terms=None)
            offset, length = int(row["genomic_offset"]), int(row["genomic_length"])
        else:
            rng = np.random.default_rng([args.seed, zlib.crc32(element.encode())])
            sequence, offset = scaffold(row[columns["sequence"]], rng)
            output = clients[row["model"]].predict_sequence(
                sequence=sequence, organism=organism, requested_outputs=output_types,
                ontology_terms=None)
            length = len(row[columns["sequence"]])
        values, tracks = summarise(output, requested, offset, length)
        metadata = {name: getattr(output, name.lower()).metadata for name in requested}
        return {"id": element, "model": row["model"], "offset": offset, "length": length,
                "values": values, "tracks": tracks, "metadata": metadata}

    tracks_path = args.out_dir / "tracks.tsv"
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool, partial_path.open("a") as partial:
        futures = {pool.submit(predict, row): row[columns["id"]] for _, row in todo.iterrows()}
        for count, future in enumerate(as_completed(futures), start=1):
            element = futures[future]
            try:
                record = future.result()
            except Exception as error:  # the other elements' predictions are kept when one fails
                failed[element] = f"{type(error).__name__}: {error}"
                print(f"  {element} failed: {failed[element]}", flush=True)
                continue
            metadata = record.pop("metadata")
            if not tracks_path.exists():
                pd.concat([frame.assign(output_type=name) for name, frame in metadata.items()],
                          ignore_index=True).to_csv(tracks_path, sep="\t", index=False)
            partial.write(json.dumps(record) + "\n")
            partial.flush()
            done[element] = record
            if count % 25 == 0 or count == len(futures):
                print(f"  {count}/{len(futures)} predicted", flush=True)

    if failed:
        print(f"{len(failed)} elements failed; run again to retry them: {sorted(failed)}",
              file=sys.stderr)
        return 1

    reference = done[elements[columns["id"]].iloc[0]]
    first = reference["tracks"]
    changed = [element for element, record in done.items() if record["tracks"] != first]
    if changed:
        print(f"The track lists differ between predictions, e.g. for {changed[:3]}.", file=sys.stderr)
        return 1
    order = list(reference["values"])
    table = pd.DataFrame([done[element]["values"] for element in elements[columns["id"]]],
                         columns=order)
    table.insert(0, columns["id"], elements[columns["id"]].to_numpy())
    table.to_csv(args.out_dir / "summaries.tsv.gz", sep="\t", index=False)
    parts = pd.Series(order).str.split("|", expand=True)
    pd.DataFrame({"column": order, "output_type": parts[0], "track": parts[1],
                  "strand": parts[2], "stat": parts[3]}).to_csv(
        args.out_dir / "columns.tsv", sep="\t", index=False)
    pd.DataFrame({
        columns["id"]: elements[columns["id"]],
        "model": [done[element]["model"] for element in elements[columns["id"]]],
        "folds": elements["folds"],
        "input_start": elements["window_start"] if args.context == "native" else "",
        "element_offset": [done[element]["offset"] for element in elements[columns["id"]]],
        "element_length": [done[element]["length"] for element in elements[columns["id"]]],
    }).to_csv(args.out_dir / "elements.tsv", sep="\t", index=False)
    provenance = {
        **settings,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "elements_predicted": len(elements),
        "models_used": elements["model"].value_counts().sort_index().to_dict(),
        "summary_columns": len(order),
        "tracks_by_output": {name: len(keys) for name, keys in first.items()},
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    partial_path.unlink()
    settings_path.unlink()
    print(f"Written to {args.out_dir}: summaries.tsv.gz ({len(elements)} x {len(order)}), "
          f"columns.tsv, tracks.tsv, elements.tsv, provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
