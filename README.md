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

C3 is implemented and available as `zimshelf.py --support`. It **annotates and
never drops a claim** — see "C3 — and why it annotates rather than gates" below
for the measurements that forced that choice.

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
- `synth.device_synth` asks the **device's own LLM** and is no longer a mock.
  `seam.mock_grounded` stays, because a device-free loop is what keeps the tests
  hermetic.

Only the *grounding* (C1) is final as-is — it is mechanical and runs anywhere.
Everything else here is a stand-in for the NPU until the embeddings run finishes.

## Layout

```
ground_answer.py   the verifier: claims + retrieved pages -> grounded | abstain
                   plus annotate_support(), which adds C3 verdicts WITHOUT gating
seam.py            the loop over an fts corpus: retrieve -> synthesize -> ground
corpus.py          the loop over the NPU vector index: two-stage cosine + the same grounding
zimshelf.py        the same loop over the offline ZIM archives, via
                   DuckDB's zim extension -- `--mode vector` is the one that
                   answers questions; `--mode fts` needs no index and no device
synth.py           the device's LLM, answering in claims the grounding layer can check
c3.py              C3 support: does the cited span actually STATE the claim?
                   cordexa's double judge, run through cordexa's own check()
c3_batch.py        re-runnable C3 characterization -- the numbers in MEASUREMENT-c3.md
c3_cards.py        C3 over a batch of cards from another system, as a file handoff
c3_rerun.py        the same, N times per card, returning the union of passes + a tally
continuity.py      locator stability across a rebuild
locator.py         naming a passage in any format, and whether it can be read back
cordexa_home.py    finding the cordexa checkout this layer borrows from
CONTRACT.md        the answer-grounding contract
MEASUREMENT-c3.md  what C3 actually does on this device, measured three times
test_*.py          hermetic unless named otherwise (no device, no network)
test_corpus.py     integration; skipped unless the vector index is present
test_c3.py         needs the device; skips loudly without it
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

# the ZIM shelf. vector mode answers questions; fts needs no index and no device.
# NOTE two different counts: `--list` shows the ARCHIVES present on disk (14),
# while the vector index spans 19 SOURCES -- some were embedded from archives
# that are no longer on the shelf, and a question can be answered from any of them.
CORDEXA_HOME=~/Projects/cordexa/main python3 zimshelf.py --list
CORDEXA_HOME=~/Projects/cordexa/main python3 zimshelf.py \
    "why does drying preserve food"

# ...and with C3 support annotation (needs the device's 30B, ~6-9s per claim)
CORDEXA_HOME=~/Projects/cordexa/main python3 zimshelf.py --support \
    "why does drying preserve food"
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
| `foo.html#methods` | id or heading text | `doc_section`, then `doc_container` |
| `any.docx#duck-blocks:12-40` | block range | `duck_blocks_slice(…, 12, 40)` |
| `any.docx#🦆:12-40` | the same, in duck | ditto |
| `notes.md#L2-L5` | line range | `read_lines(src)`, filtered on `line_number` |
| `zim://wiki.zim/Photosynthesis#intro` | duckeye's zim scheme + section | as HTML |
| `plain.txt` | the whole document | `panduck_read_blocks(src)` |

**A named fragment resolves on two axes.** `doc_section` walks *headings* and bounds on
heading level — the prose under a title. `doc_container` (new in panduck) walks the
*structural* nesting — what is inside a `<div>`, `<section>`, list or blockquote. Neither
subsumes the other: a div can hold three headings, and a heading's section can run across
several divs. A fragment on `<h2 id="methods">` wants the first; one on
`<div id="sidebar">` wants the second, and `doc_section` returns nothing at all for it.
So `#name` tries section, then container — which is closer to what a browser does with a
fragment than either alone.

Whether a `#name` resolves at all also depends on the **installed webbed**, not on
panduck: the published build keeps `id` only on div, section/article and headings, so
`<ul id="steps">` is unaddressable today. `locator.ID_CAPTURE` records that.

**The block range is namespaced on purpose.** `#p.27` and `#L2-L5` are established
spellings that predate us, so a document is unlikely to carry an id shaped like
either. A block range is *our* invention, and a bare `#b12-b40` is a plausible real
element id — numbered boxes, lettered clauses, generated HTML. Namespacing means a
document that genuinely contains `<div id="b12-b40">` still resolves as a section.
`#🦆:…` is an **alias**, not a second canonical form: `render()` emits
`duck-blocks:` so stored locators stay greppable and compare equal. The
percent-encoded duck parses to the same locator.

`#duck-blocks:…` is the format-agnostic one: every `duck_block` carries `element_order` whatever the
reader was, so a block range addresses HTML, Markdown, DOCX, EPUB and PDF identically.
Pages exist only in paginated formats, sections only where there are headings. Prefer the
most specific form the format supports; fall back to `#b…`, which always exists.

**Lines resolve through a different shape.** `read_lines(src)` — the community extension
duckeye already installs for its `-f lines` mode — yields `line_number, content,
byte_offset`, so `#L2-L5` is a `BETWEEN` on `line_number`. It returns *text lines*, not
`duck_block`s, unlike every other resolver. That is fine for grounding: C1 checks that a
span occurs verbatim in the text of the thing cited, and the content of lines 2–5 is
exactly that text. Cite `#b<range>` instead if you need blocks. Lines are addressed
*beside* the block model rather than through it, which is why searching panduck for line
support finds nothing.

A citation may be **the string itself** — a model writes `manual.pdf#p.27`, not a dict —
and `ground_answer` parses it into the structured identity the comparison uses.

## Depends on

- **cordexa** (`~/Projects/cordexa`) — imports its mechanical C1 (`reference/c1_span.py`)
  and extends its `spec/retrieval.md`. If tiibrarian's verifier ever lands upstream,
  it lands in cordexa. Set `CORDEXA_HOME` if cordexa is elsewhere.
- **duckdb** ≥ 1.5.5 (python) for retrieval.

Pairs with `tiiny-duckdb-rag` (the retrieval/ask loop) and `TTt` (device tooling).

## C3 — and why it annotates rather than gates

The `support_checked: False` gap is closed. `c3.py` runs cordexa's double judge
against the device's 30B, and `ground_answer.annotate_support(result, check_kept)`
attaches the verdicts to a grounded answer. `zimshelf.py --support` wires it up:

```bash
CORDEXA_HOME=~/Projects/cordexa/main python3 zimshelf.py --support \
    "how do you dry food without a dehydrator"
```

**It annotates. It never drops a claim**, and that is a measured decision rather
than caution. Across three characterization runs C3 produced **0 false accepts
out of 3 adverse cases every time**, and repeating the adverse fixtures gave
12/12 `not supported` over **24/24 individual askings — not one split**. But a
borderline *genuine* paraphrase is unstable: the same span and claim at
temperature 0 returned `supported` on only **4 of 6 consecutive runs**, because
the two askings honestly disagree and `combine` resolves a split conservatively.

So the instability manufactures false negatives and never false positives. A
gate would therefore delete true claims at that rate, and the failure would be
invisible — the answer loses a correct sentence and still reads clean. A claim
is marked; a reader, or a later gate with repeats or a better model, decides.
`test_support_annotation.py` pins this and is mutation-checked: an
implementation that filters `supported: False` fails it.

`support_checked: False` still means **no verdict was reached** — device down,
gateway cut, no ANSWER line, or the served model changed between askings. It
never means "not supported". See MEASUREMENT-c3.md for all three runs, including
a correction: "recall 2/3" was never a property of one fixture, it is run-to-run
variance, and an n=6 single run cannot distinguish 2/3 from 3/3.

### Handing C3 to another system

`c3_cards.py` takes `{card_id, claim, span, content_sha256}` and returns a
verdict per card; `c3_rerun.py` does N passes and returns the union of passes
with a `k/3` tally. The `content_sha256` is echoed back **unexamined** — it is a
token the caller issued and verifies on ingest, so this side cannot assert that
a span is the one the caller grounded on.

The device call is deliberately **not** injectable. Four guards live in it that
are specific to this appliance: the served-model identity cross-check (a chat
completion here reports a generic `../model/model.q8_0.gguf` whatever is loaded,
so cordexa's own identity test is weak and `c3_batch.py` compensates against the
management API), the ~60s bodiless-504 gateway budget, retry-transport-only so a
reply that *arrived* is never re-asked, and the unparseable-but-complete →
unmeasured override. An injected `ask` loses all four, silently.

The full loop runs on the device: embed the question on the NPU → two-stage cosine
over the page vectors → the device's LLM writes claims that quote and cite → mechanical
grounding drops anything not verbatim on the page cited → C3 annotates what survives
→ answer, or abstain.
