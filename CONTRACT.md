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
| verbatim in the CITED page | `grounded` | kept |
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

### A citation must resolve to exactly one page

`_same_page` compares only the identity fields a citation and a page *share* —
deliberately loose, so extra fields (scores, urls) never block a match. But a
citation carrying only the generic fields (say `{kind, id}` with no
`path`/`page`) then agrees with **every** retrieved page. Bound to whichever
page was retrieved first, such a claim reads as `grounded` whenever its span
happens to sit on that page — a free pass for precisely the misattribution this
layer exists to catch. An LLM emitting a partial citation is the expected case,
not an exotic one.

So a citation must resolve to **exactly one** retrieved page:

| matches | meaning | disposition |
|---|---|---|
| 1 | the answer named a page | check the span against it |
| 0 | cited nothing that was retrieved | no cited page — cannot be `grounded` |
| ≥2 | names a *family* of pages, not a page | under-specified; not attribution — cannot be `grounded` |

Mechanical, no tunable, consistent with C1 being the only mechanical check. A
claim whose citation does not resolve falls through to the existing
`misattributed` / `unfounded` verdicts; every verdict carries `citation_matched`
so the reason is inspectable.

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
