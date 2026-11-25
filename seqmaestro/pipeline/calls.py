"""How many calls each arm of a run makes to the language model, the tokens they carry, what they
cost and how long they wait, estimated before the run starts (`--estimate`).

**The calls are counted from the run's settings**, part by part (`count`):
- a written source's proposal is `proposal_rounds` requests, and each request round whose features
  fail on the rows sends one repair, up to `repair_rounds` times; the iterative source makes
  `1 + feedback_rounds` proposals in each cell;
- DEFT's generative finder makes, at each node, `1 + 2R` prompts for `S` candidates each (`R` its
  reflections, `S` its candidates per prompt), sent as `ceil(S / per_request)` requests, and one
  request for each candidate's code, with a repair for each candidate whose code fails. A tree has
  as many nodes as the tree the time estimate grows on the run's rows under DEFT's settings at the
  low end, and `2^depth - 1` at the high end (`seqmaestro.pipeline.vendored.deft.call_settings`);
- IGSR runs its agent `successors` times at each node it expands, each run making one to three
  calls, and expands between `ceil(budget / successors)` and as many nodes as its budget allows
  (`seqmaestro.pipeline.vendored.igsr.call_settings`);
- the judge, where the grid runs it, rates each of the `top_n` features of every rated cell
  `repeats` times, one call each, and the recurrence analysis makes one call for each arm on a
  written source and one across them.

Every count is a range: the low end with no repair and the fewest nodes, the high end with every
repair and the most nodes.

**Each call is sized from what it sends.** The proposer's first request is rendered on the run's
training rows, as a cell renders it, and counted with the model's tokeniser
(`litellm.token_counter`). A request of DEFT's carries the same instructions with one example row,
and IGSR's the same examples as the proposer's. A reply is counted in written features: a proposal
asks for `n_per_round`, a DEFT candidate is one (its description and its code), an IGSR call one.
How many tokens one feature takes, and how fast the model writes, are measured by two small calls
(`measure`): one feature written to a short request, whose tokens give the size of a feature, and a
one-word reply, whose time is the latency of a call, so that the feature's time beyond it gives the
speed of the reply. **Dollars are priced call by call**, since a provider may price a call's tokens by
the call's size. Both go
through the run's client with its `language_model` settings, so a model that reasons spends what it
would in the run.

**Dollars** are the tokens priced from LiteLLM's list (`litellm.cost_per_token`), for the model the
environment names, or for Azure OpenAI's base model, before any discount the provider gives for a
prompt it has cached, which can be much of DEFT's. A model the list does not hold is not priced.

**Waiting** is each call's latency plus its reply's tokens at the measured speed. A process makes
one call at a time, except that DEFT asks for the code of a prompt's `S` candidates at once, while
its requests for candidates go one after another. **The load** is what the run asks of the provider
at once: the calls in flight when every slot for a process that calls a language model is taken
(`--llm-slots`), and the requests and tokens per minute those processes make while they wait.

**Without the language model's credentials** nothing is measured or priced, and the estimate says
so; the counts and the requests' sizes are still given.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from seqmaestro.arms import DEFT_OR_CART, IGSR, arm_name, written
from seqmaestro.features import LLM_GENERATED, LLM_ITERATIVE, LLM_JOINT, parse_pool
from seqmaestro.judge.engine import REPLY_TOKENS as JUDGE_REPLY_TOKENS
from seqmaestro.llm import LiteLLMClient, describe_calls, load_task, missing_settings, provider_from_environment
from seqmaestro.llm.columns import describe_columns
from seqmaestro.pipeline.repository import DESCRIPTIONS
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.vendored import deft, igsr
from seqmaestro.proposer.prompt import SYSTEM_PROMPT, render_request, sample_examples
from seqmaestro.report import naming

#: The call that times a reply of almost nothing, which is a call's latency.
PING = ("Reply with one word.", "Reply with the word ready.", 256)
#: The call that has the model write one feature, whose reply gives the tokens of a written feature
#: and, with the latency, the speed of a reply.
ONE_FEATURE = (
    "You write features of DNA sequences as Python code.",
    "Write one feature of a DNA sequence that could help to predict whether a protein binds it. "
    "Reply with JSON holding `name`, `description`, `rationale` and `code`, where `code` defines "
    "`def feature(sequence):` returning a number.",
    2000)

#: The slots for processes that call a language model at which the load is shown, beside the run's.
SLOTS = (1, 2, 5, 10, 20)

#: How many characters of a feature's description a recurrence request shows (the analysis cuts
#: each description there).
RECURRENCE_ITEM_CHARACTERS = 300


@dataclass(frozen=True)
class Speed:
    """What the two timed calls measured."""

    #: The seconds of a call with almost no reply.
    latency: float
    #: The reply's tokens per second.
    tokens_per_second: float
    #: The tokens of one feature's reply.
    feature_tokens: int
    #: What the two calls cost, where the model is priced.
    cost: float | None

    def seconds(self, reply_tokens: float) -> float:
        """How long a call with this reply takes."""
        return self.latency + reply_tokens / self.tokens_per_second


@dataclass(frozen=True)
class Kind:
    """One kind of call a part makes: how many over the run (at the low end and the high end), the
    tokens each sends, its reply (in written features and in tokens besides), and how many of these
    calls a process makes at once (`together`)."""

    low: int
    high: int
    sends: int
    features: float = 0.0
    tokens: int = 0
    together: int = 1

    def reply(self, speed: Speed) -> float:
        return self.features * speed.feature_tokens + self.tokens


@dataclass(frozen=True)
class Part:
    """The calls of one arm or analysis over the run, made by `processes` processes."""

    name: str
    kinds: tuple[Kind, ...]
    processes: int

    @property
    def in_flight(self) -> int:
        """The most calls one of its processes makes at once."""
        return max(kind.together for kind in self.kinds)

    def calls(self) -> tuple[int, int]:
        return sum(kind.low for kind in self.kinds), sum(kind.high for kind in self.kinds)

    def sent(self) -> tuple[int, int]:
        return (sum(kind.low * kind.sends for kind in self.kinds),
                sum(kind.high * kind.sends for kind in self.kinds))

    def replied(self, speed: Speed) -> tuple[float, float]:
        return (sum(kind.low * kind.reply(speed) for kind in self.kinds),
                sum(kind.high * kind.reply(speed) for kind in self.kinds))

    def waiting(self, speed: Speed) -> tuple[float, float]:
        """The seconds its processes wait on the model, summed over them, at each end: each kind's
        calls take their time, divided by how many of them go at once."""
        return (sum(kind.low * speed.seconds(kind.reply(speed)) / kind.together
                    for kind in self.kinds),
                sum(kind.high * speed.seconds(kind.reply(speed)) / kind.together
                    for kind in self.kinds))


# ------------------------------------------------------------------------------- sizing


@dataclass(frozen=True)
class Sizes:
    """The tokens of what a pool's written source sends: the proposer's first request on the run's
    training rows, the same instructions with no example row, and one example row on average."""

    proposal: int
    instructions: int
    example: float
    #: The tokens of one item of a recurrence request: a description cut where the analysis cuts it.
    item: int


def tokens(text: str, model: str) -> int:
    """The tokens of `text` in the model's tokeniser, as LiteLLM counts them."""
    import litellm

    return int(litellm.token_counter(model=model, text=text))


def sizes(config, rows: Rows, pool: str, model: str) -> Sizes:
    """Render the proposer's first request for a pool's cells, on the training rows, and count it."""
    dataset = rows.dataset
    inputs = dataset.inputs(rows.train)
    y = np.asarray(rows.train[dataset.target_column])
    proposer = config.sources.proposer
    context = load_task(DESCRIPTIONS, config.dataset.task)
    columns = describe_columns(inputs, dataset.sequence_column, dataset.covariates_for(pool))
    examples = sample_examples(inputs, y, proposer.n_examples, config.grid.seeds[0],
                               dataset.target_kind)

    def request(shown):
        return SYSTEM_PROMPT + "\n" + render_request(
            context, proposer.n_per_round, shown, [], dataset.sequence_column,
            dataset.target_column, dataset.target_kind, columns=columns,
            max_example_length=config.grid.generated_features.max_example_length,
            guidance=config.grid.generated_features.guidance)

    proposal, instructions = tokens(request(examples), model), tokens(request([]), model)
    item = tokens(context.render()[:RECURRENCE_ITEM_CHARACTERS], model)
    return Sizes(proposal=proposal, instructions=instructions,
                 example=(proposal - instructions) / max(len(examples), 1), item=item)


# ------------------------------------------------------------------------------- counting


def arm_part(model: str, pool: str, config, size: Sizes, provider: str | None,
             nodes: int | None = None) -> Part | None:
    """An arm's calls over the run (the module docstring says how each is counted), or None for an
    arm that calls no language model. `nodes` is the splits of the tree the time estimate grew on
    the run's rows, which DEFT's tree is counted at, at the low end."""
    members = parse_pool(pool)
    cells = len(config.grid.labels) * len(config.grid.seeds)
    proposer, generated = config.sources.proposer, config.grid.generated_features
    name = arm_name(model, pool)
    if LLM_ITERATIVE in members or LLM_GENERATED in members:
        proposals = cells * ((1 + config.sources.feedback_rounds) if LLM_ITERATIVE in members else 1)
        requests = proposals * proposer.proposal_rounds
        return Part(name, (
            Kind(requests, requests, size.proposal, features=proposer.n_per_round),
            Kind(0, requests * generated.repair_rounds, size.proposal, features=proposer.n_per_round),
        ), processes=cells)
    if LLM_JOINT in members and model == DEFT_OR_CART:
        found = deft.call_settings(config, provider)
        prompts = 1 + 2 * found["reflections"]
        candidates = found["candidates"]
        most = 2 ** found["max_depth"] - 1
        fewest = min(nodes, most) if nodes else most
        asks = math.ceil(candidates / found["per_request"])
        per_node = prompts * asks, prompts * candidates
        sends = size.instructions + round(size.example)
        # A candidate's reply is one feature: its description in the semantic reply, and its code.
        # The requests for candidates go one after another, and the code of a prompt's candidates
        # is asked for at once.
        return Part(name, (
            Kind(cells * fewest * per_node[0], cells * most * per_node[0], sends,
                 features=0.5 * candidates / asks),
            Kind(cells * fewest * per_node[1], cells * most * per_node[1], sends, features=0.5,
                 together=candidates),
            Kind(0, cells * most * per_node[1] * generated.repair_rounds, sends, features=0.5,
                 together=candidates),
        ), processes=cells)
    if LLM_JOINT in members and model == IGSR:
        found = igsr.call_settings(config)
        budget, successors = found["budget"], found["successors"]
        fewest = math.ceil(budget / successors)
        most = max(m for m in range(fewest, budget + 1)
                   if m + (successors - 1) * math.ceil((m - 1) / successors) <= budget)
        return Part(name, (Kind(cells * successors * fewest, cells * successors * most * 3,
                                size.proposal, features=1.0),), processes=cells)
    return None


def analysis_parts(config, size: Sizes) -> list[Part]:
    """The calls of the judge, where the grid runs it, and of the recurrence analysis, where an arm
    writes features."""
    analyses, cells = config.analyses, len(config.grid.seeds)
    parts = []
    if config.grid.judge:
        rated = len(config.arm_cells()) * cells * len(set(analyses.judge.labels) & set(config.grid.labels))
        calls = rated * analyses.top_n * analyses.judge.repeats
        parts.append(Part("judge", (Kind(calls, calls, size.instructions, tokens=JUDGE_REPLY_TOKENS),),
                          processes=1))
    writing = [arm for arm in config.arm_cells() if written(arm[1])]
    if writing:
        items = analyses.top_n * cells * len(config.grid.labels)
        kinds = [Kind(len(writing), len(writing), size.instructions + items * size.item,
                      tokens=4 * items + 24 * math.ceil(items / 6))]
        if analyses.recurrence.across_arms and len(writing) > 1:
            every = items * len(writing)
            kinds.append(Kind(1, 1, size.instructions + every * size.item,
                              tokens=4 * every + 24 * math.ceil(every / 6)))
        parts.append(Part("recurrence", tuple(kinds), processes=1))
    return parts


# ------------------------------------------------------------------------------- measuring


def pricing_model(record: dict[str, Any]) -> str:
    """The model LiteLLM prices a run's calls as: Azure OpenAI's base model, or the model called."""
    return str(record.get("base_model") or record["model"])


def price(model: str, sent: float, replied: float) -> float | None:
    """What `sent` and `replied` tokens cost in dollars on LiteLLM's list, or None where the list does
    not hold the model."""
    import litellm

    # LiteLLM prints its list of providers on every model it does not know.
    litellm.suppress_debug_info = True
    try:
        prompt, completion = litellm.cost_per_token(model=model, prompt_tokens=int(sent),
                                                    completion_tokens=int(replied))
    except Exception:
        # LiteLLM raises a bare Exception for a model its list does not hold.
        return None
    return float(prompt + completion)


def measure(client, model: str, clock=time.perf_counter) -> Speed:
    """Time the two calls of the module docstring through the run's client: the feature first, and
    then the one-word reply, so that what the first call spends opening the connection adds to the
    feature's time, which makes the speed lower and the waiting longer, and not to the latency."""
    started = clock()
    written_ = client.complete(*ONE_FEATURE[:2], max_completion_tokens=ONE_FEATURE[2])
    seconds = clock() - started
    feature = client.completion_tokens
    started = clock()
    pinged = client.complete(*PING[:2], max_completion_tokens=PING[2])
    latency = clock() - started
    for completion in (written_, pinged):
        if not completion.ok:
            raise RuntimeError(f"A call to time the language model failed: {completion.error}")
    if feature <= 0:
        raise RuntimeError("The language model reported no tokens for its reply, so its speed "
                           "cannot be measured.")
    return Speed(latency=latency, tokens_per_second=feature / max(seconds - latency, 0.1 * seconds),
                 feature_tokens=feature,
                 cost=price(model, client.prompt_tokens, client.completion_tokens))


# ------------------------------------------------------------------------------- the load


def load(parts: list[Part], speed: Speed, slots: int) -> dict[str, float]:
    """What the arms ask of the provider with `slots` processes calling it at once: the calls in
    flight when every slot is taken by the processes that keep the most going, the requests and
    tokens per minute at that point, and the hours the arms' calls take at the high end. The
    analyses run after the arms, one process each, so they do not add to the peak."""
    arms = [part for part in parts if part.name not in ("judge", "recurrence")]
    processes = sorted(((part.in_flight, part) for part in arms for _ in range(part.processes)),
                       key=lambda item: -item[0])[:slots]
    in_flight = sum(count for count, _ in processes)
    per_minute = tokens_per_minute = 0.0
    for _, part in processes:
        waiting = part.waiting(speed)[1]
        if not waiting:
            continue
        per_minute += 60 * part.calls()[1] / waiting
        tokens_per_minute += 60 * (part.sent()[1] + part.replied(speed)[1]) / waiting
    work = [part.waiting(speed)[1] / part.processes for part in arms for _ in range(part.processes)]
    hours = max(sum(work) / slots, max(work, default=0.0)) / 3600
    return {"slots": slots, "in_flight": in_flight, "requests_per_minute": per_minute,
            "tokens_per_minute": tokens_per_minute, "hours": hours}


# ------------------------------------------------------------------------------- the estimate


@dataclass
class Estimate:
    """The language model's part of a run's estimate."""

    parts: list[Part]
    slots: int
    speed: Speed | None = None
    #: The model the calls are priced as, where the environment names one.
    model: str | None = None
    #: Why the speed and the cost were not measured, where they were not.
    unmeasured: str | None = None
    sizes: dict[str, Sizes] = field(default_factory=dict)

    def cost(self, part: Part) -> tuple[float | None, float | None]:
        """A part's cost at each end, priced call by call, because a provider may price a call's
        tokens by the call's size."""
        if self.speed is None or self.model is None:
            return None, None
        each = [price(self.model, kind.sends, kind.reply(self.speed)) for kind in part.kinds]
        if None in each:
            return None, None
        return (sum(kind.low * one for kind, one in zip(part.kinds, each, strict=True)),
                sum(kind.high * one for kind, one in zip(part.kinds, each, strict=True)))


def estimate(config, slots: int, nodes: dict[str, int] | None = None,
             client_factory=LiteLLMClient) -> Estimate:
    """Count, size, time and price the run's calls to the language model. `nodes` holds, by pool,
    the splits of DEFT's tree as the time estimate grew it."""
    pools = [pool for _, pool in config.arm_cells() if written(pool)]
    if not pools and not config.grid.judge:
        return Estimate(parts=[], slots=slots)
    missing = missing_settings()
    provider = None if missing else provider_from_environment()
    record = None if provider is None else describe_calls(provider, config.language_model)
    model = None if record is None else pricing_model(record)
    rows = Rows.of(config)
    found = {pool: sizes(config, rows, pool, model or "") for pool in dict.fromkeys(pools or
                                                                                  [config.arm_cells()[0][1]])}
    parts = [part for arm in config.arm_cells()
             if (part := arm_part(*arm, config, found.get(arm[1], next(iter(found.values()))),
                                  provider.name if provider else None,
                                  (nodes or {}).get(arm[1]))) is not None]
    parts += analysis_parts(config, next(iter(found.values())))
    if missing:
        return Estimate(parts=parts, slots=slots, sizes=found,
                        unmeasured="the language model's settings are not in the environment ("
                                   + "; ".join(missing) + ")")
    try:
        speed = measure(client_factory(config.language_model), model)
    except RuntimeError as error:
        return Estimate(parts=parts, slots=slots, sizes=found, model=model, unmeasured=str(error))
    return Estimate(parts=parts, slots=slots, speed=speed, model=model, sizes=found)


# ------------------------------------------------------------------------------- reporting


#: What a figure that could not be estimated is shown as.
MISSING = "\N{EN DASH}"


def _span(low: float | None, high: float | None, show) -> str:
    if low is None or high is None:
        return MISSING
    return show(low) if round(low, 6) == round(high, 6) else f"{show(low)} to {show(high)}"


def _dollars(value: float) -> str:
    return f"${value:,.2f}" if value >= 0.01 else "<$0.01"


def _millions(value: float) -> str:
    return f"{value / 1e6:.2f} M" if value >= 1e5 else f"{value / 1e3:.0f} k"


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 360 else f"{seconds / 60:.0f} min"


def table(found: Estimate) -> str:
    """The language model's part of the estimate, as the driver prints it."""
    if not found.parts:
        return "No part of this run calls a language model."
    lines = []
    if found.speed is None:
        lines.append(f"The calls are counted and sized; their cost and waiting are not estimated, "
                     f"because {found.unmeasured}.")
    else:
        speed = found.speed
        lines.append(
            f"The language model ({found.model}) answers after {speed.latency:.1f} s and writes "
            f"{speed.tokens_per_second:.0f} tokens a second; one written feature is "
            f"{speed.feature_tokens} tokens. The two calls that measured this cost "
            f"{_dollars(speed.cost) if speed.cost is not None else 'an amount LiteLLM cannot price'}.")
    lines.append(f"{'arm or analysis':42} {'calls':>17} {'tokens sent':>18} {'tokens replied':>18} "
                 f"{'cost':>18} {'waiting':>16}")
    for part in found.parts:
        sent = part.sent()
        replied = part.replied(found.speed) if found.speed else (None, None)
        waiting = part.waiting(found.speed) if found.speed else (None, None)
        lines.append(f"{part.name:42} {_span(*part.calls(), str):>17} {_span(*sent, _millions):>18} "
                     f"{_span(*replied, _millions):>18} {_span(*found.cost(part), _dollars):>18} "
                     f"{_span(*waiting, _hours):>16}")
    costs = [found.cost(part) for part in found.parts]
    if found.speed is not None and all(cost[1] is not None for cost in costs):
        lines.append(f"All calls: {_span(sum(c[0] for c in costs), sum(c[1] for c in costs), _dollars)}.")
        lines.append("\nThe load on the provider, by how many processes may call it at once "
                     "(`--llm-slots`). Fewer slots spread the same calls over a longer time, so the "
                     "cost stays the same and the run takes longer.")
        lines.append(f"{'slots':>7} {'calls in flight':>16} {'requests a minute':>18} "
                     f"{'tokens a minute':>16} {"the arms' calls take, at most":>31}")
        for slots in sorted({*SLOTS, found.slots}):
            row = load(found.parts, found.speed, slots)
            mark = "  (this run)" if slots == found.slots else ""
            lines.append(f"{slots:>7} {row['in_flight']:>16} {row['requests_per_minute']:>18,.0f} "
                         f"{row['tokens_per_minute']:>16,.0f} {_hours(3600 * row['hours']):>31}{mark}")
    if any(part.in_flight > 1 for part in found.parts):
        lines.append("A DEFT tree is counted at the nodes of the tree the time estimate grew on these "
                     "rows at the low end, and at the most its depth allows at the high end.")
    if found.speed is not None:
        lines.append("The cost is before any discount the provider gives for a prompt it has cached.")
    return "\n".join(lines)


def as_record(found: Estimate) -> dict[str, Any]:
    """The language model's part of the estimate, for the run's `logs/estimate.json`."""
    return {
        "model": found.model,
        "unmeasured": found.unmeasured,
        "speed": None if found.speed is None else vars(found.speed),
        "parts": [{"name": part.name, "calls": part.calls(), "sent": part.sent(),
                   "replied": part.replied(found.speed) if found.speed else None,
                   "cost": found.cost(part),
                   "waiting_seconds": part.waiting(found.speed) if found.speed else None,
                   "processes": part.processes, "in_flight": part.in_flight}
                  for part in found.parts],
        "load": ([load(found.parts, found.speed, slots) for slots in sorted({*SLOTS, found.slots})]
                 if found.speed else None),
    }


def grid(config, wall: dict[str, float], found: Estimate) -> str:
    """The grid a user decides on before the launch, as a markdown table: the pools down, the models
    across, and in each arm's cell its time (`wall`, by arm, the seconds it would take with the
    machine to itself) and, where it calls a language model, its cost and its waiting. Each pool
    and model is named as the report names it, with its key in the config beside it."""
    models = list(dict.fromkeys(model for model, _ in config.arm_cells()))
    parts = {part.name: part for part in found.parts}
    header = [f"{naming.MODELS[model]['short']} (`{model}`)" for model in models]
    lines = ["| Pool | " + " | ".join(header) + " |", "|---|" + "---|" * len(models)]
    for pool, entry in config.arms.items():
        cells = []
        for model in models:
            name = arm_name(model, pool)
            if model not in entry.models:
                cells.append("")
                continue
            shown = _hours(wall[name]) if name in wall else MISSING
            part = parts.get(name)
            if part is not None and found.speed is not None:
                shown += (f", {_span(*found.cost(part), _dollars)}, "
                          f"{_span(*part.waiting(found.speed), _hours)} waiting")
            elif part is not None:
                shown += f", {_span(*part.calls(), str)} calls"
            cells.append(shown)
        label = naming.parse(arm_name(entry.models[0], pool)).source_short.capitalize()
        lines.append(f"| {label} (`{pool}`) | " + " | ".join(cells) + " |")
    return "\n".join(lines)
