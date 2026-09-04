# tiibrarian

**The Tiiny's librarian: offline answers that cite a real page, or say they don't know.**

The [Tiiny AI Pocket Lab](https://tiiny.ai) can turn a shelf of scanned manuals
into a knowledge base you ask by meaning — embed the query, search millions of
page-vectors, and let a local model write a cited answer. The danger is the
citations: a nearest-neighbour index *always returns pages*, so a fluent answer
can rest on pages that do not contain it, and the `[1]` looks the same either way.

tiibrarian is the layer that makes those citations mean something. It applies
cordexa's rule — **retrieval proposes, C1 decides** — at *answer* time:

> Every claim in the answer must quote a span. Each span is checked, mechanically,
> against the page the claim **cites**. A span that isn't verbatim on the cited
> page is dropped. If nothing survives, the librarian **abstains** — "the library
> doesn't cover this" — instead of confabulating.

## What "grounded" means — and does not

`grounded` means exactly: *these words are verbatim on the page cited.* It does
**not** mean the claim is true, or even that the span supports it. A verbatim,
correctly-cited span can still be over-reached (page says "drying", claim says
"sun drying"). That is the **C3** question — claim-vs-span support — and it needs
the model. tiibrarian keeps such a claim, flagged `support_checked: False`, and
never renders "grounded" to a reader as "true." See `CONTRACT.md`.

## The NPU is the final substrate — this is not optional

The production librarian runs **on the NPU**: the query is embedded on-device,
cosine search runs over the on-device vector index, and synthesis is the
device's local LLM. That is the whole point of the Pocket — meaning in, meaning
out, nothing leaves the device.

What's in this repo today is **NPU-free scaffolding**, built deliberately so the
whole loop is runnable and testable on a workstation *while the device is busy
building the vector index*:

- `seam.retrieve()` uses DuckDB **fts** over a docs corpus. The final version
  swaps it for **cosine over the on-device survivorlibrary vectors** — same seam,
  different `retrieve()`.
- `seam.synth` is a **mock** that emits quoted claims. The final version points
  it at the device's local `/v1` endpoint (the `tiiny-duckdb-rag` ask shape).

Only the *grounding* (C1) is final as-is — it is mechanical and runs anywhere.
Everything else here is a stand-in for the NPU until the embeddings run finishes.

## Layout

```
ground_answer.py   the verifier: claims + retrieved pages -> grounded | abstain
seam.py            the loop: retrieve -> synthesize -> ground  (retrieval is real; synth pluggable)
CONTRACT.md        the answer-grounding contract
test_*.py          hermetic (no device, no network) — real fts retrieval + mechanical C1
```

## Run

```bash
# tests — no device, no network
CORDEXA_HOME=~/Projects/cordexa python3 -m unittest discover -p 'test_*.py'

# a grounded ask over a docs corpus (mock synth; real retrieval)
CORDEXA_HOME=~/Projects/cordexa python3 seam.py "what is duckdb" \
    --db ~/Projects/tiiny-duckdb-rag/corpus.duckdb --topk 3
```

## Depends on

- **cordexa** (`~/Projects/cordexa`) — imports its mechanical C1 (`reference/c1_span.py`)
  and extends its `spec/retrieval.md`. If tiibrarian's verifier ever lands upstream,
  it lands in cordexa. Set `CORDEXA_HOME` if cordexa is elsewhere.
- **duckdb** ≥ 1.5.5 (python) for retrieval.

Pairs with `tiiny-duckdb-rag` (the retrieval/ask loop) and `TTt` (device tooling).

## When the device frees

First device run is **C3 support over the answer claims** — does the grounded
span actually support the spoken claim — which closes the `support_checked: False`
gap. Then swap `retrieve()` to the vector backend and `synth` to the on-device
LLM, and tiibrarian is answering from the real library, on the NPU.
