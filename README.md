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

`grounded` means exactly: *these words are verbatim on every page the citation names.* It does
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

- `seam.retrieve()` uses DuckDB **fts** over a docs corpus — kept, because it is real
  retrieval that needs no device and makes the tests hermetic.
- `corpus.retrieve()` is the **vector backend, and it exists now**: two-stage cosine over
  the SurvivorLibrary page vectors embedded on the NPU. Same seam, different `retrieve()`,
  as planned. Verified against a real 1,310,336-page index.
- `seam.synth` is still a **mock** that emits quoted claims. The final version points it
  at the device's local `/v1` endpoint (the `tiiny-duckdb-rag` ask shape). This is now the
  only piece standing between the loop and a real answer.

Only the *grounding* (C1) is final as-is — it is mechanical and runs anywhere.
Everything else here is a stand-in for the NPU until the embeddings run finishes.

## Layout

```
ground_answer.py   the verifier: claims + retrieved pages -> grounded | abstain
seam.py            the loop over an fts corpus: retrieve -> synthesize -> ground
corpus.py          the loop over the NPU vector index: two-stage cosine + the same grounding
locator.py         naming a passage in any format, and whether it can be read back
CONTRACT.md        the answer-grounding contract
test_ground_answer.py, test_seam.py   hermetic (no device, no network)
test_corpus.py     integration; skipped unless the vector index is present
```

## Run

```bash
# tests — no device, no network
CORDEXA_HOME=~/Projects/cordexa python3 -m unittest discover -p 'test_*.py'

# a grounded ask over a docs corpus (mock synth; real fts retrieval)
CORDEXA_HOME=~/Projects/cordexa python3 seam.py "what is duckdb" \
    --db ~/Projects/tiiny-duckdb-rag/corpus.duckdb --topk 3

# a grounded ask over the SurvivorLibrary vector index (needs the device to embed
# the question: TIINY_AUTH_KEY + the NPU embedder)
CORDEXA_HOME=~/Projects/cordexa TIINY_AUTH_KEY=... python3 corpus.py \
    "how do you can green beans without a pressure canner"
```

**The query must be embedded by the same encoder that built the index** — the device's
NPU embedder, not its CPU one. Same model, different backends: cosine between their
outputs is ~0.97, and the CPU's are L2-normalised while the NPU's are raw. Mixing them
does not fail, it just quietly retrieves worse, so `corpus.embed_query` raises rather
than substituting anything.

## Locators — citing a passage in any format

The corpus is not only PDFs, so a citation cannot only be a page number. A locator is a
document plus a fragment naming a position inside it, and **every form maps to a call
that can read it back** — a citation that cannot be resolved is decoration.

| locator | means | read back by |
|---|---|---|
| `manual.pdf#p.27` | page (or `#p.3-7`) | `read_pdf_blocks(src, pages := '27')` |
| `foo.html#methods` | section, by heading id **or** text | `doc_section(src, 'methods')` |
| `any.docx#b12-b40` | block range | `duck_blocks_slice(…, 12, 40)` |
| `notes.md#L2-L5` | line range | **nothing yet — see below** |
| `zim://wiki.zim/Photosynthesis#intro` | duckeye's zim scheme + section | as HTML |
| `plain.txt` | the whole document | `panduck_read_blocks(src)` |

`#b…` is the format-agnostic one: every `duck_block` carries `element_order` whatever the
reader was, so a block range addresses HTML, Markdown, DOCX, EPUB and PDF identically.
Pages exist only in paginated formats, sections only where there are headings. Prefer the
most specific form the format supports; fall back to `#b…`, which always exists.

**`#L2-L5` parses and round-trips but `locator.resolvable()` returns False**, because
nothing in the stack addresses lines: panduck has no line-range reader and `duck_block`s
carry no line numbers. It is the obvious spelling for Markdown and a person will write it,
so it is accepted and recorded — and honestly reported as unresolvable rather than
silently treated as a section name.

A citation may be **the string itself** — a model writes `manual.pdf#p.27`, not a dict —
and `ground_answer` parses it into the structured identity the comparison uses.

## Depends on

- **cordexa** (`~/Projects/cordexa`) — imports its mechanical C1 (`reference/c1_span.py`)
  and extends its `spec/retrieval.md`. If tiibrarian's verifier ever lands upstream,
  it lands in cordexa. Set `CORDEXA_HOME` if cordexa is elsewhere.
- **duckdb** ≥ 1.5.5 (python) for retrieval.

Pairs with `tiiny-duckdb-rag` (the retrieval/ask loop) and `TTt` (device tooling).

## When the device frees

`retrieve()` is done — `corpus.py` is the vector backend, tested against the real index.
Two things remain:

1. **C3 support over the answer claims** — does the grounded span actually support the
   spoken claim — which closes the `support_checked: False` gap.
2. **`synth` against the on-device LLM**, replacing the mock.

With those, tiibrarian is answering from the real library, on the NPU.
