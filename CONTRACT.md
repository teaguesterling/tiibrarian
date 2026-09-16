# Answer-time grounding — the contract

`spec/retrieval.md` (cordexa) applied to a **synthesized answer**, for the pocket
appliance. Prototype: `ground_answer.py` (+ `test_ground_answer.py`). NPU-free.

## The problem it exists for

The appliance retrieves pages by cosine similarity and a local model writes an
answer with `[n]` citations. Cordexa's retrieval doctrine, verbatim:

> Retrieval proposes candidates. C1 decides. A nearest-neighbour index returns a
> neighbour for every query, including queries with no answer — it is
> structurally incapable of returning nothing.

So a confident answer can rest on pages that do not contain it, and the field
report's own Q&A is candid that its answers are *"synthetic, retrieval-shape."*
This layer makes them defensible.

## The rule (mechanical, no tunable)

Every claim in the answer must **quote a span**, and each span is checked against
the page the claim **cites**:

| span is… | verdict | disposition |
|---|---|---|
| verbatim on **every** page the citation names | `grounded` | kept |
| verbatim on **some but not all** of them | `ambiguous` | dropped |
| verbatim in a DIFFERENT retrieved page | `misattributed` | dropped |
| in NO retrieved page | `unfounded` | dropped |
| a span C1 **declines to judge** | `undecidable` | dropped |
| verbatim, but in a page that is **not the document requested** | `drifted` | dropped |

**Abstain, structurally:** any non-`grounded` claim is dropped; the answer
abstains iff **zero** claims survive. No ratio, no threshold — consistent with C1
being the only mechanical check. A coverage floor would be a heuristic and does
not belong here (checks.md's line).

### `undecidable` — dropped, but not accused

C1 returns **three** verdicts, not two: `pass`, `fail`, and `abstain`. `abstain`
means the check *cannot decide this input* — today, a span carrying letters the
Latin fold cannot compare (Greek, Cyrillic), which in a corpus of chemistry,
medicine and mathematics is an ordinary page, not an edge case.

This layer collapsed `abstain` into "not found" until 2026-09-05, and so reported
`unfounded` — an accusation of confabulation — for spans that were genuinely on
the page. Measured, before the fix:

| span | C1 | reported |
|---|---|---|
| plain Latin, really present | `pass` | `grounded` |
| contains Greek, really present | `abstain` | `unfounded` ← wrong |
| truly not in the document | `fail` | `unfounded` |

Two things were wrong with the middle row. It states, to a reader, that the model
made something up when the checker said only that it could not look. And it
inflates the paraphrase rate this project reports about its own device runs,
because a claim dropped by the checker's limit was counted against the model.

`undecidable` is therefore its own verdict, carrying C1's reason verbatim. It is
still **dropped** — declining to decide is not grounding — but it is not a
finding against the answer. The distinction is load-bearing for anyone reading
`dropped` as a quality measurement.

**Open question for cordexa:** whether `abstain` should propagate as its own
state through a caller's pipeline, or whether a caller is expected to drop it.
This layer chose to propagate and drop; the choice is recorded here rather than
assumed to be the spec's.

### `drifted` — C1.8, and why this layer has to check it itself

`spec/checks.md` C1.8: *"real span from document B, filed under requested locator
A — fail."* Ask for `A/Mint`, get `A/Spearmint` back, quote a sentence that really
is in Spearmint: every word is true and the record is broken, because the answer
is about a different plant. Cordexa's note on that case is blunt about the stakes
— it is *"the failure that shipped 10 of 25 rows elsewhere."*

`check()` catches it structurally: it compares the claimed locator against
`retrieved`, and when no locator is given it defaults the claim to the document's
`requested`. Drift is then a locator mismatch.

**This layer overrides that default on every call.** `_c1_verdicts` passes
`cited_locator=p["retrieved"]`, deliberately — attribution here is computed, not
taken from the model, so the question asked of C1 is "are these words on *this*
page". But `retrieved` always equals `retrieved`, so the locator arm becomes a
tautology and C1.8 stops being checked at all.

It was not checked, from the first version of this file until 2026-09-05, when
running cordexa's golden set surfaced it: `c09` is `expect: 0` and this layer
returned `grounded`. `_drift` now re-establishes the comparison the substitution
removed. Only a `pass` is downgraded — a span that is absent is `unfounded`
whether the page drifted or not, and hiding that behind a retrieval complaint
would be its own mislabelling.

Note what the defect was: not a wrong rule, but a **check that silently stopped
running** because a caller changed an argument the check's own default was doing
work in. Nothing failed. Every test stayed green.

Today `corpus.py` and `seam.py` set `requested == retrieved` by construction, so
their pipelines cannot drift and the guard is inert for them. That is a property
of the current retrievers, not a guarantee — the moment this layer consumes a real
adapter (ZIM, or panduck's `doc_*`), drift becomes reachable.

## What a pass means — and does not

`grounded` means exactly: **these words are verbatim on the page cited.** It does
**not** mean the span supports the claim. A verbatim, correctly-cited span can
still be over-reached by its claim — the `ch2z_031` shape: page says "drying",
claim says "sun drying". C1 cannot see that; the layer keeps such a claim and
marks it `support_checked: False`. **Claim-vs-span support is the C3 question and
needs the device.** Never render `grounded` to a user as "true".

## Attribution is the prize while the NPU is down

C1 alone says the span is in *some* page. The stronger mechanical fact —
computed here via `c1_span`'s `cited_locator` / `_same_locator` — is that the
span is in **the page the answer cited**. A span that is real but on page 114
while the answer says page 27 is the defect nobody catches in a *spoken* answer.
That is `misattributed`, and it is a first-class dropped verdict.

### A citation names pages; ambiguity matters only when it changes the verdict

`_same_page` compares only the identity fields a citation and a page *share* —
deliberately loose, so extra fields (scores, urls) never block a match. A citation
therefore resolves to a **set** of retrieved pages, which may be empty, one, or several.

An earlier version of this contract required **exactly one** and refused otherwise. That
was too strict, and it rejected true claims. The corpus contains passages that appear
verbatim at more than one position — measured: the same text at p.57 of two printings of
one chemistry manual, and 1.8% of pages are exact duplicates of another. A citation that
cannot tell two identical passages apart is still a **correct** citation: the span is
verbatim on both, so the verdict is the same whichever was meant.

The rule is therefore about **consequence, not arity**:

| citation resolves to | span is verbatim on | verdict |
|---|---|---|
| 0 pages | — | not cited; falls through to `misattributed` / `unfounded` |
| N ≥ 1 pages | **all N** | `grounded` — the ambiguity cannot change the answer |
| N > 1 pages | **some, not all** | `ambiguous` — binding to one would be arbitrary *and* outcome-changing |
| N ≥ 1 pages | **none** | falls through to `misattributed` / `unfounded` |

With N = 1 this is exactly the old behaviour. Every verdict carries `citation_matched`,
and a `grounded` claim with N > 1 also carries `found_on` listing the pages it is true of,
so a reader can see that the citation was imprecise even though the claim stands.

**This is why locators should be granular.** The way to avoid `ambiguous` is not a looser
check but a citation that names a position precisely enough to pin one passage —
`{kind, source, book, page}` today, growing to chapter, section, or a passage ordinal as
passages become sub-page. `source` is already load-bearing: 21,726 (book, page) pairs
exist in both the native-text and re-OCR lanes, so a locator without it is not unique.

## When the device frees

The first device run is **C3 support over answer claims** against these fixtures
(does the grounded span actually support the spoken claim), NOT the old
window-sweep suite. That closes the `support_checked: False` gap and completes
the grounded-answer path: retrieve → synthesize → C1 cited-page grounding
(here, mechanical) → C3 support (device) → answer or abstain.

## What this layer assumes about C1, and where that is written down

`ground_answer.py` imports cordexa's `reference/c1_span.py` **by filesystem path**,
via `CORDEXA_HOME`. That file describes itself as a reference implementation, not
the normative spec: it may change, there is no version pin, and a replacement that
kept the signature would change verdicts on real answers without raising anything.
Every other test in this repo asserts *through* C1, so all of them would still pass.

`test_c1_surface.py` exists to make that coupling visible. It asserts, in one place
and by name, only the properties this layer would be **wrong** without:

- `check(span, document, cited_locator=...) -> (verdict, reason)`, `cited_locator`
  passed by keyword.
- The verdict vocabulary is exactly `{pass, fail, abstain}`, all three reachable,
  and `abstain` distinct from `fail` — see `undecidable` above.
- `text` is read from the document on every call: the same span against two
  different documents gives two different verdicts. `locate_span`'s whole claim to
  compute attribution rests on this.
- The locator rule is `requested ⊆ claimed`, every claimed key equals the
  **retrieved** value, and extra fields on `retrieved` do not block. If C1 stopped
  comparing against `retrieved`, `misattributed` would stop being detectable — and
  that is the defect this layer exists to catch.
- A pass means whole tokens, and a length floor exists. The floor is read from
  `MIN_SIGNIFICANT_CHARS` rather than written as a literal: *that* it binds is the
  contract, *where* it sits is cordexa's to move.

Verified by mutation, 2026-09-05: five single-line changes to `c1_span.py` — abstain
folded into fail, comparing `requested` instead of `retrieved`, dropping the subset
rule, dropping the token-boundary check, and zeroing the floor — each fail this
suite, and the unmutated file passes. Guards that cannot fail are not guards.

### Conformance, and what is still not covered

An earlier draft of this section said conformance was unverifiable here "because
that needs cordexa's harness and golden cases, which this repo does not have."
That was asserted without looking. Cordexa ships `examples/golden/` — 58 cases
over 33 documents, each tagged with the check that should decide it and the star
level the pipeline must land on — plus `reference/test_c1_span.py`. Both are
readable from `CORDEXA_HOME`, and the second failure mode this section exists to
prevent is exactly that one: treating an unread thing as known.

`test_conformance.py` runs this layer against the 13 C1-target cases, read-only —
the golden set's README reserves editing to the instrument owner, and this repo
does not touch it. It skips cleanly when the set is absent.

Results, 2026-09-05: **13/13 agree with cordexa on disposition** (kept vs dropped),
after the `drifted` fix that this run produced. On three cases (`c08`, `f03`,
`f06`) this layer's *label* is narrower than C1's: C1 folds "you cited a document
the span is not in" into one `fail`, and this layer separates `misattributed` from
`unfounded`, because only the first is repairable. Same disposition, more
information — a refinement, not a disagreement.

**Still not covered.** C2 (numeric, entity) is not implemented in this repo at
all. C3 (support) is `c3.py` and needs the device. The star levels are a
whole-pipeline judgement, so agreeing on 13 C1 cases is not a claim that the
appliance conforms to `checks.md` — and the golden set's own README says a score
against it "is a score against a test fixture, not against a corpus."

## Home

Prototype lives here (consumer side), standalone-importable via `CORDEXA_HOME`.
If it lands upstream it is **cordexa** — it uses cordexa's C1 and extends its own
`retrieval.md`, not anything grimoire-specific.
