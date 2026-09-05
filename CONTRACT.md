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

**Abstain, structurally:** any non-`grounded` claim is dropped; the answer
abstains iff **zero** claims survive. No ratio, no threshold — consistent with C1
being the only mechanical check. A coverage floor would be a heuristic and does
not belong here (checks.md's line).

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

## Home

Prototype lives here (consumer side), standalone-importable via `CORDEXA_HOME`.
If it lands upstream it is **cordexa** — it uses cordexa's C1 and extends its own
`retrieval.md`, not anything grimoire-specific.
