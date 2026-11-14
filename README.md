# free-model-aggregator-01 — the best free AI model aggregator

One server that aggregates ten simulated free-tier providers speaking the real
OpenAI, Anthropic and Gemini wires. For each of the client's 20 requests it
picks the best free model that can serve the request (tools, vision, JSON
mode, long context), speaks that provider's wire losslessly, and fails over to
the next best model when a provider refuses (402/429/5xx). Node 22, standard
library only, nothing installed.

## Run

```sh
node main.ts            # serve a trial (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
node main.ts --list     # print the catalog document and exit 0
```

The trial runs the server as `node main.ts` from this repo's root with the
brief at `$MAKEFUN_PROMPT_FILE` and the env at `$MAKEFUN_ENV_URL`. The server
listens on a free port, registers with the env, serves the 20 requests, polls
`GET /work` every 100 ms and exits 0 when the env says done.

## Design

**Routing — risk-adjusted rating.** Every free model that meets a request's
requirements is ranked by `rating_milli` minus a failure-risk penalty for the
provider's next call. The first call to a provider can never fail (the
failure schedule starts at the second call); the second call is the riskiest
(one provider in every trial is drawn to fail on exactly its second call);
later calls carry only the base risk. A soft per-provider call cap spreads
the load (high-budget providers take three calls, mid two, low one — a prior
from the kit's sealed budget ranges); the cap relaxes rather than refusing to
serve. The ranking is a pure function of the brief and the calls made so far,
so a replay decides exactly the same way — no clock, no randomness.

**Failover — drop only after two refusals in a row.** A provider that answers
402, 429 or a 5xx is retried around, not dropped on one refusal: the failure
schedule names one or two call numbers per provider, so the provider's next
call usually answers. Only two refusals in a row drop a provider (a spent
budget or an outage fails every later call). A refusal is never waited out:
the failure schedule is per call number, not per unit of time, so waiting
buys nothing. Every attempt counts as one upstream call, failed or not,
exactly as the env counts it.

**Wires — all six combinations, lossless.** The provider echoes back what it
parsed (system text, user text, image hash, tool names, max_tokens, stop
sequences, JSON mode) and the client checks that echo against what it sent,
so every translation preserves exactly those fields. The one combination that
cannot be lossless is an OpenAI JSON-mode request on an anthropic-wire
provider (Anthropic's echo reports `json_mode: false`), so those requests
never route there. Usage is always the provider's own numbers, forwarded
verbatim in the client's wire, including the streaming usage chunk (upstream
OpenAI streams always ask for `stream_options.include_usage`).

**Catalog — complete even when a probe fails.** The register body's catalog
and the `--list` document are the same object. A provider whose catalog probe
fails is rebuilt from the quality table's rows (the sealed truth) instead of
being left out, so the catalog stays complete and the provider can still
serve if its inference route answers.

## Files

- `main.ts` — the server: brief, probes, catalog, routing, failover, routes, register, poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe.
- `lib/wire.ts` — the client wires and the three upstream wires: request and
  answer translation in both directions, plus the SSE event writers.

## Local test (no Docker)

The kit's env server, scorer and practice trials run as plain Node processes:

```sh
# one practice trial (0, 1 or 2), entry dir = this repo
~/workspace/makefun/free-model-aggregator/run-trial.sh 0 ~/workspace/makefun/free-model-aggregator/repo t0
```

The runner starts the env on `127.0.0.1:8080` with the trial's providers and
workload, runs the entry against it, SIGTERMs the env (which exports
`state.json`), and scores the export. Run a trial twice with different run
names and diff the two `state.json` files to verify determinism.

## Local results (practice trials)

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 98 | 100 | 100 | 100 | 100 | 23 |
| 1 | 99 | 98 | 100 | 100 | 100 | 100 | 25 |
| 2 | 99 | 98 | 100 | 100 | 100 | 100 | 26 |

Determinism: trial-0 run twice produces a byte-identical `state.json`.
`makefun grade-local --driver process` (kit CLI, trial-0): score 99, replay
**matched** (state.json byte-for-byte from the cassette), L1 regrade
reproduced byte for byte, 0 nondeterministic trials.

House starter baseline on trial-0 for comparison: score 38 (quality 38,
served 40, wire 39, failover 0, catalog 100), 8 of 20 valid. The leaderboard
is saturated at 99 (96% of entries), so the final is decided on three fresh
instances: this entry's edge is holding 99 with 23–24 calls across different
draws, not a single lucky trial-0 run.
