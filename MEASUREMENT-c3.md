# C3 support over answer claims — characterization (2026-09-05)

## Third run, 2026-10-04 — and the recall figure is not a property of the case

Re-run of the same six fixtures while wiring `--support` into `ask_grounded`,
with the 30B and the embedder co-resident but no bulk job running (the woollama
pool gate landed that day, so interactive calls queue instead of colliding):

**measured 6/6 · false-accepts 0/3 · recall 3/3.** All twelve askings served by
one value; 0/12 elaboration cuts; 4.8–8.1s per case.

G2 — the paraphrase case that missed in *both* earlier runs, and which the "Next"
section below therefore called the place "where the appliance's answers will feel
it" — passed. G3 passed here too, but had just come back **not supported** from
the identical span and claim inside the `annotate_support` end-to-end check,
minutes earlier.

So the same input gave two different verdicts. That was worth measuring rather
than explaining away, and the measurement is the finding:

| G3, identical span + claim, temperature=0, six consecutive runs |
|---|
| supported, supported, supported, supported, **not supported**, **not supported** |

**4/6 supported.** In both flips, ordering A flipped and ordering B held — the
double judge disagreed and `combine` resolved the disagreement conservatively.

Three consequences:

1. **"recall 2/3" was never a property of G2.** It is run-to-run variance on
   whichever borderline case the dice land on, and an n=6 single run cannot tell
   2/3 from 3/3. The three runs recorded here are 2/3, 2/3, 3/3 — consistent with
   one unstable case out of three genuine ones, not with a specific weak case.
   Any future rate needs repeats per case, not more cases.
2. **The instability is in the safe direction.** `combine` resolves a split to
   *not supported*, so a flip manufactures a false negative and never a false
   positive. 0/3 false-accepts has held across all three runs. This is precisely
   why the wiring annotates and does not gate: the unstable direction is the one
   that would silently delete true claims from an answer.
3. **A C3 verdict is not idempotent, and anything that re-runs one must say what
   happens on a flip.** A consumer that replaces a stored verdict on re-run can
   demote on the second look with no new evidence. Promotion on `supported` is
   safe; *demotion* on a later `not supported` is not, because the later verdict
   is not better-informed than the earlier one.

**The adverse cases were then repeated too, and they do not flip.** O1
(over-reach) and R1 (refuted numeral), six runs each on identical input:
12/12 `not supported`, and — the part that matters — **24/24 individual
askings** said not supported. Not one split, not one near-miss.

So the asymmetry is not merely an artefact of `combine` resolving splits
conservatively. The adverse cases are stably separated; the instability is
specific to borderline genuine paraphrase, where the two askings genuinely
disagree. Cumulative adverse observations across all runs: 21, zero false
accepts.

This is what makes accumulating passes across repeated runs safe. A consumer
that re-runs a batch and keeps every `supported` ever seen is ratcheting — and
a ratchet ratchets errors too, permanently. That would be reckless on a
0-out-of-9 bound (rule of three puts the true rate as high as ~33%). It is
defensible here because the adverse cases show no instability to accumulate,
not because the observed count is zero.

Still not established: the flip rate for clear-cut *genuine* cases (G1 has
never flipped but has not been repeated), whether the longer latency on the two
flipped runs (8.7s, 8.6s vs ~6.7s) means anything, and whether paraphrase
instability varies by domain. n=6 per case, three cases repeated.


## Second run, 13:29 — through cordexa's `check()`, in a serialized NPU window

Reproducible: `c3_batch.py`, run in a serialized NPU window. **Reconstructed fixtures**
(see below) — comparable in shape to the 09:43 run, not the same six cases.

| case | expected | A | B | verdict | ok |
|---|---|---|---|---|---|
| G1 verbatim-genuine | supported | supported | supported | promoted | ✅ |
| G2 paraphrase-genuine | supported | not supported | supported | held (disagree) | ⚠️ recall miss |
| G3 verbatim-drying | supported | supported | supported | promoted | ✅ |
| O1 over-reach (ch2z_031) | not supported | not supported | not supported | caught | ✅ |
| O2 specialization | not supported | not supported | not supported | caught | ✅ |
| R1 refuted numeral | not supported | not supported | not supported | caught | ✅ |

**measured 6/6 · false-accepts 0/3 · recall 2/3.** 90 seconds wall-clock, 6–9s
per case — an order of magnitude cheaper than the 60s-per-asking budget assumed.

What the guards recovered from `check()` actually reported, none of which the
first run could see:

- **`served`: one value across all twelve askings.** The identity assumption is
  now a recorded fact rather than a reconstruction from the watchdog log.
- **`elaboration_cut` 0/12, `reversed_in_elaboration` 0/12** — no asking was cut,
  and no whole asking reversed itself after committing.
- **`committed_early` False on all 12.** v3 exists to make the model reason
  before it commits; this is the first evidence that it does, on this model at
  this budget. Under the old code the flag was never computed.

**A caveat that only appeared by looking.** The served string is
`../model/model.q8_0.gguf` — a generic path, not a model id. The management API
names the same model `Qwen/Qwen3-30B-A3B-Instruct`. So cordexa's identity check,
which compares what the *completion* returns between askings, is **weak on this
device**: two different chat models could both report that path and pass. It is
still worth having (it catches the proxy reporting a change) but it is not the
guarantee it looks like. `c3_batch.py` now also records the management API's
`models/running` before and after the run and flags a change — the strong version
of the same check, and the one to read.

**Against the 09:43 run:** identical headline numbers, and G2's paraphrase miss
reproduces — that cost is real, not a one-off. O2 changed character: it was caught
by *disagreement* then (one ordering fooled) and by **both orderings rejecting**
now. That is a stronger result, but O2's text is reconstructed, so it is not
evidence the model improved — only that this phrasing is easier.

---

## First run, 09:43 — the original characterization

First device run of `c3.py` (cordexa's support-check v3 double-judge, on the Tiiny
`Qwen3-30B-A3B-Instruct`, no frontier model). **A demonstration, not a measurement
— n=6.** Raw: `c3_batch_results.json`.

> **The script that produced this table was never versioned.** It left the results
> json and a log on the workstation and nothing else, so these numbers could be read
> but not reproduced. `c3_batch.py` now holds a re-runnable set: same six roles,
> over page text that really is in this repo's fixtures, but **reconstructed** —
> G3 was a radio manual whose text was not kept anywhere. A re-run is comparable
> in shape and is not the same six cases; report it as its own run.

| case | expected | A | B | verdict | ok |
|---|---|---|---|---|---|
| G1 verbatim-genuine | supported | supported | supported | promoted | ✅ |
| G2 paraphrase-genuine | supported | not supported | supported | held (disagree) | ⚠️ recall miss |
| G3 verbatim-radio | supported | supported | supported | promoted | ✅ |
| O1 over-reach (ch2z_031) | not supported | not supported | not supported | caught | ✅ |
| O2 specialization (indirect) | not supported | not supported | supported | held (disagree) | ✅ |
| R1 refuted numeral (30 vs 60) | not supported | not supported | not supported | caught | ✅ |

**false-accepts 0/3 · recall 2/3.**

## What it establishes

- The motivating case works: the **over-reach C1 cannot see is caught by C3** (O1,
  both orderings agree "not supported" — the claim adds "sun" the passage doesn't
  state). This is the specialization the earlier *naive* single-ask judge
  rubber-stamped 44/45; the disciplined double-judge gets it.
- **Precision-first holds:** 0 false-accepts across over-reach, specialization, and
  a refuted numeral.

## What it does NOT establish (honest limits)

- **n=6.** Not a rate. Real numbers need the full answer-fixture / perturbation
  suite re-run as *answer* claims.
- **Recall costs paraphrase.** G2 ("prevents microbial growth" for "bacteria, yeast
  and mold cannot grow") is a fair paraphrase, held only because the two orderings
  disagreed. For a spoken answer this means the librarian will sometimes abstain on
  a *correct* paraphrased answer — a UX cost, not a correctness bug. cordexa's model:
  a held claim stays at a lower star and is recovered by re-phrasing / re-running.
- **Disagreement is carrying weight.** O2 (specialization) was caught by
  *disagreement*, not by both orderings rejecting — one ordering was fooled. A
  determined over-reach that fools *both* orderings would pass. O1's both-agree
  rejection is the stronger result; how often over-reach reaches both-agree vs
  disagree is unmeasured.

- **The run predates the served-model identity check.** These six cases were
  measured by the first `c3.py`, which reimplemented cordexa's `check()` loop and
  discarded the model string the device returns with each reply. Cordexa's
  `check()` refuses a verdict whose two askings were served by *different* models
  — "the run has no single identity" — and that refusal was not running here.

  It is not a paper risk here: the local watchdog log for that day records a
  model stop-and-restart at 12:05:17, well after this run.

  `c3_batch_results.json` carries no timestamps and no `served` field — the
  evidence that would settle it from the inside was never recorded. But the
  watchdog log is an independent record, and it does log exactly this event: the
  12:05 lines are the C3 model being stopped by name. **The run's file is dated
  09:43:25, and the log shows no model start or stop between 07:34:15 and
  12:05:17.** Six cases at two askings, each bounded by the ~60s gateway cap, is a
  window of at most ~12 minutes — well inside that quiet period.

  So the assumption behind the table — **one model answered all twelve askings** —
  is not merely plausible, it is checked, against an instrument that demonstrably
  records the event when it happens. The limit worth stating: the watchdog logs
  the swaps *it* causes. A model changed by anything else — another session, the
  device on its own — would not appear there. Strong evidence, not proof.

  `c3.py` now calls cordexa's `check()` directly, so a re-run records `served` per
  asking and refuses a two-model verdict outright, and this reconstruction stops
  being necessary. **The 13:29 re-run did exactly that** — see the top of this
  file: one served value across all twelve askings, recorded rather than inferred.

## Next

**Turn the demonstration into a rate.** Re-run over the full grounded-answer
fixture set and the perturbation specializations as *answer* claims. Watch the
paraphrase-recall cost specifically — G2 has now missed in both runs, so that is
where the appliance's answers will feel it, and it is the number worth having.

Report `trace_flag_line` beside the rate. A rate with no account of how many
askings were cut or committed early is a number without its method.

**Serializing the NPU works; keep doing it.** The run used a local wrapper that
pauses the embedder's watchdog and the embed driver, runs the batch, and
restores both from a trap on exit — worth writing for whatever else shares your
NPU. Measured on 2026-09-05: a 90-second window, and the embed job resumed with
nothing lost —
staged 2,418,577, shards 1454, `pending` back to ~0. Do not run C3 alongside the
embedder: the 12:05 event above is this project's own instance of sustained
concurrent NPU load, where the 30B was resident beside the embedder, the
embedder stopped responding, and recovery had to stop both — 48s against the 6s
a start-only recovery takes.

**Fixtures.** The six cases in `c3_batch.py` are reconstructed. If the original
spans and claims turn up, replace them and say so; otherwise the honest framing
is that these are this repo's fixtures, and the 09:43 table is a separate run.

**Open, for cordexa.** The served-model identity check compares what the
completion returns between askings, and on this device that value is a generic
`../model/model.q8_0.gguf` for any loaded chat model. The guard is not wrong, but
it is weaker than it reads, and a caller cannot tell from the value alone. Worth
knowing on their side; `c3_batch.py` compensates locally by recording the
management API's `models/running` before and after.
