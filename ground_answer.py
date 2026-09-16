"""Answer-time grounding — `spec/retrieval.md` applied to a synthesized answer.

The pocket appliance retrieves pages by meaning and a local model writes an
answer with `[n]` citations. Cosine retrieval "is structurally incapable of
returning nothing" (retrieval.md), so a confident answer can rest on pages that
do not contain it. This layer applies cordexa's rule — **retrieval proposes,
C1 decides** — to the answer: every claim must quote a span, and each span is
checked, mechanically, against the page the claim cites.

WHAT A PASS MEANS, EXACTLY (and no more): the quoted span is verbatim in the
page the answer cited. That is C1's whole competence here. It does **not** mean
the span supports the claim — a verbatim, correctly-cited span can still be
over-reached by its claim (the `ch2z_031` shape: page says "drying", claim says
"sun drying"). Claim-vs-span support is the C3 question and needs the device;
this layer marks it `support_checked: False` and never calls a grounded claim
"true". "Grounded" == "these words are really on that page", full stop.

A citation that resolves to ZERO retrieved pages is not attribution, and the claim
cannot be grounded on it. Resolving to several is handled below -- it is a failure
only when the ambiguity changes the verdict.

ATTRIBUTION IS COMPUTED, NOT TRUSTED. We know which pages were shown and what was
quoted, so where a quote came from is a fact -- `locate_span` establishes it. A claim
that arrives WITHOUT a citation gets a derived one and cannot be misattributed, which is
why that is the preferred shape. A claim that arrives WITH one is still checked against
where the span really is, because a model that asserts a provenance can assert a false
one, and that disagreement is worth surfacing rather than silently overwriting.

A citation resolves to the retrieved pages whose identity it matches. Resolving to
SEVERAL is not itself a failure: if the span is verbatim on all of them the answer is
the same whichever was meant, so the claim stands. It fails only when the ambiguity is
outcome-changing -- the span on some candidates and not others -- which is `ambiguous`.

The mechanical verdicts, per the advisor's split:
  - span in the CITED page            -> grounded            (kept)
  - span in a DIFFERENT retrieved page -> misattributed       (dropped)  <- the spoken-answer defect
  - span in NO retrieved page          -> unfounded           (dropped)  <- confabulation
  - cited pages disagree about the span -> ambiguous          (dropped)  <- cannot tell which
  - C1 DECLINES to judge the span      -> undecidable         (dropped)  <- not an accusation
  - span real, but the page is NOT the document requested -> drifted (dropped)  <- C1.8

`undecidable` is the fourth because C1 has THREE verdicts, not two: `abstain`
means the check cannot decide this input, not that the span is absent. Collapsing
it into `unfounded` told a reader the model confabulated when the checker had only
declined to look — and this corpus is full of the Greek that triggers it. Dropped
either way; accused in only one. See CONTRACT.md.

The properties of C1 this module relies on are asserted, by name, in
`test_c1_surface.py` — the import is by filesystem path, against a file that
declares itself non-normative, so the coupling is written down where a change to
it breaks a test that says so.

Abstain rule, structural (stated before any output, no tunable): a claim that is
not `grounded` is dropped; the answer ABSTAINS iff zero claims survive. A
coverage floor would be a heuristic and belongs on the far side of
checks.md's mechanical/heuristic line — not here.

NPU-free: C1 is mechanical. Only the synthesis that produced the claims used the
device. Import cordexa's C1 by path so the home of this module stays open.
"""
import importlib.util
import os

import cordexa_home

CORDEXA = cordexa_home.find()


def _load_c1():
    path = os.path.join(CORDEXA, "reference", "c1_span.py")
    spec = importlib.util.spec_from_file_location("cordexa_c1_span", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_C1 = None


def _c1():
    global _C1
    if _C1 is None:
        _C1 = _load_c1()
    return _C1


def _loc(d):
    return d.get("path") or d.get("url") or d.get("title") or d.get("doc") or str(d)


def _drift(page):
    """-> a reason if this page is not the document that was REQUESTED, else "".

    C1.8, `spec/checks.md`: "real span from document B, filed under requested
    locator A -> fail". A retrieval that asks for Mint and returns Spearmint has
    a real span in a real document that answers a different question, and
    cordexa's note on that case says it is "the failure that shipped 10 of 25
    rows elsewhere".

    We have to check this ourselves, and that is not optional. `check()` compares
    the claimed locator against `retrieved`, and defaults the claim to the
    document's `requested` — which is exactly how it catches drift. This layer
    passes `cited_locator=p["retrieved"]` on every call, because it computes
    attribution rather than trusting it. That substitution makes the locator arm
    a tautology (retrieved always equals retrieved) and so DISABLES C1.8. Found
    2026-09-05 by running cordexa's golden set: `c09` is `expect: 0`, and this
    layer returned `grounded`.
    """
    req, ret = page.get("requested"), page.get("retrieved")
    if not isinstance(req, dict) or not isinstance(ret, dict):
        return ""
    off = sorted(k for k, v in req.items() if k not in ret or ret[k] != v)
    if not off:
        return ""
    return ("retrieval drift: %s was requested and %s came back (differs on %s, "
            "resolution=%s) — a real span in the wrong document"
            % (_loc(req), _loc(ret), ", ".join(off), page.get("resolution")))


def _c1_verdicts(span, pages):
    """[(page, verdict, reason)] for every retrieved page. C1 has THREE verdicts.

    `pass`, `fail` and `abstain`, and abstain is not a soft fail -- c1_span.py is
    explicit that it means the check CANNOT DECIDE this input (today: a span carrying
    letters the Latin fold cannot compare, such as Greek or Cyrillic). Collapsing it
    into `fail` reports "the model made this up" when the checker said "I cannot judge
    this", which is an accusation the check never made -- and this corpus is full of
    chemistry, medicine and mathematics, so it fires on real pages.
    """
    c1 = _c1()
    out = []
    for p in pages:
        v, why = c1.check(span, p, cited_locator=p["retrieved"])
        if v == "pass":
            # Only a pass is downgraded. A span that is not on the page is not on
            # it whether the page drifted or not, and saying "drift" there would
            # hide a plain unfounded claim behind a retrieval complaint.
            drift = _drift(p)
            if drift:
                v, why = "drift", drift
        out.append((p, v, why))
    return out


def locate_span(span, pages):
    """Every retrieved page the span is verbatim on. Attribution, computed.

    We know which pages were shown to the model and we know what it quoted, so where the
    quote came from is a FACT we can establish -- not something the model has to be
    trusted to report. Asking it to cite is asking for a value we can compute, and the
    only thing that can add is a mistake.

    Pages C1 ABSTAINS on are not located: it did not say the span is there.
    """
    return [p for p, v, _ in _c1_verdicts(span, pages) if v == "pass"]


def _cited_pages(pages, cited):
    """Every retrieved page the citation resolves to. May be 0, 1, or several.

    A citation that resolves to nothing is not attribution. A citation that resolves
    to SEVERAL is not automatically a failure -- see verify_claim. What matters is
    whether the ambiguity can change the verdict, not that it exists: if the span is
    verbatim on every page the citation names, binding to any one of them gives the
    same answer, and refusing would reject a claim that is true of all of them.

    Real case, measured on the SurvivorLibrary index: the same passage appears verbatim
    at p.57 of two different printings of the same chemistry manual. A citation that
    cannot tell those two apart is still a correct citation.
    """
    return [p for p in pages if _same_page(p.get("retrieved"), cited)]


def _narrow(page, span):
    """The finest locator that still contains the span. -> identity dict.

    A page whose text runs on into the next is handed to C1 JOINED, so a
    quotation crossing the break can be checked at all (corpus.hits_to_pages).
    But most spans found in a joined document sit wholly inside one of its pages,
    and citing `p.300-301` for something entirely on p.300 is a worse citation
    than the index can support.

    So: having grounded on the join, ask each constituent page whether it holds
    the span by itself, and cite that instead when exactly one does. Derived, not
    asked for -- the same rule as the rest of this module. A span that genuinely
    straddles the break is in no single part, and keeps the range.
    """
    parts = page.get("parts")
    if not parts:
        return page["retrieved"]
    c1 = _c1()
    inside = [p for p in parts
              if c1.check(span, p, cited_locator=p["retrieved"])[0] == "pass"]
    return inside[0]["retrieved"] if len(inside) == 1 else page["retrieved"]


def _dropped(claim, span, verdict, cited, n, why, derived):
    return {"claim": claim["text"], "verdict": verdict, "span": span,
            "cited_page": cited, "citation_matched": n, "derived": derived,
            "why": why}


def _undecidable(claim, span, cited, n, why, derived):
    """Dropped, but NOT accused. The checker could not judge; the model may be right."""
    return _dropped(claim, span, "undecidable", cited, n,
                    "C1 could not decide this span: " + (why or "no reason given"),
                    derived)


def verify_claim(claim, pages):
    """claim: {"text", "span", "cites"} where `cites` is the retrieved-identity
    of the page the answer attributes the span to (a page's `retrieved`).
    pages: [adapter Document]. -> a per-claim verdict dict."""
    cited = _as_ident(claim.get("cites"))
    span = claim.get("span", "")

    # Every page's verdict, computed ONCE. This is the ground truth every branch below
    # is measured against, and asking C1 twice for the same pair invites the two answers
    # to drift.
    verdicts = _c1_verdicts(span, pages)
    found = [p for p, v, _ in verdicts if v == "pass"]
    undecided = [(p, why) for p, v, why in verdicts if v == "abstain"]
    drifted = [(p, why) for p, v, why in verdicts if v == "drift"]

    if cited is None:
        # NO CITATION GIVEN -- derive it. This is the preferred path: a locator we
        # computed cannot be misattributed, so a reader following it always lands on
        # text that really contains the quote.
        if found:
            return {"claim": claim["text"], "verdict": "grounded", "span": span,
                    "cited_page": _narrow(found[0], span),
                    "found_on": [p["retrieved"] for p in found] if len(found) > 1 else None,
                    "citation_matched": len(found), "derived": True,
                    "support_checked": False, "why": ""}
        if drifted:
            # Definite, and worse than not knowing: the words are real and in a
            # document that answers a different question.
            return _dropped(claim, span, "drifted", None, 0, drifted[0][1], True)
        if undecided:
            return _undecidable(claim, span, None, 0, undecided[0][1], True)
        return {"claim": claim["text"], "verdict": "unfounded", "span": span,
                "cited_page": None, "citation_matched": 0, "derived": True,
                "why": "span does not occur verbatim in any retrieved page"}

    # 1) the pages this citation names, and whether the span is on them
    matches = _cited_pages(pages, cited)
    n = len(matches)
    if n:
        by_id = {id(p): (v, why) for p, v, why in verdicts}
        on = [p for p in matches if by_id[id(p)][0] == "pass"]
        ab = [(p, by_id[id(p)][1]) for p in matches if by_id[id(p)][0] == "abstain"]
        dr = [(p, by_id[id(p)][1]) for p in matches if by_id[id(p)][0] == "drift"]
        if len(on) == n:
            # Verbatim on EVERY page the citation names. With n == 1 this is the
            # ordinary case. With n > 1 the citation is under-specified but harmless:
            # every candidate gives this same verdict, so the ambiguity is immaterial.
            return {"claim": claim["text"], "verdict": "grounded", "span": span,
                    "cited_page": cited, "citation_matched": n,
                    "found_on": [p["retrieved"] for p in on] if n > 1 else None,
                    "support_checked": False, "why": ""}
        if on:
            # Verbatim on SOME but not all. Here the ambiguity DOES change the verdict:
            # binding to one candidate grounds, binding to another does not, and nothing
            # in the citation says which was meant. Refuse rather than pick.
            return {"claim": claim["text"], "verdict": "ambiguous", "span": span,
                    "cited_page": cited, "citation_matched": n,
                    "found_on": [p["retrieved"] for p in on],
                    "why": "the citation names %d retrieved pages and the span is "
                           "verbatim on only %d of them, so which page is cited "
                           "changes the verdict" % (n, len(on))}
        if dr:
            return _dropped(claim, span, "drifted", cited, n, dr[0][1], False)
        if ab:
            # The cited page could not be judged. Not misattribution and not
            # confabulation -- there is no finding to report against the model.
            return _undecidable(claim, span, cited, n, ab[0][1], False)
        # verbatim on none of them -> fall through

    amb = ("" if n <= 1 else
           " (the citation matched %d retrieved pages and the span is on none of "
           "them)" % n)

    # 2) not on the cited page(s) — is it verbatim in some OTHER retrieved page?
    cited_set = {id(p) for p in matches}
    for p in found:
        if id(p) in cited_set:
            continue
        return {"claim": claim["text"], "verdict": "misattributed", "span": span,
                "cited_page": cited, "found_on": p["retrieved"],
                "citation_matched": n, "derived": False,
                # The correct locator, since we know it. A caller that would rather
                # repair the citation than drop the claim has what it needs; this
                # layer still drops it, because the model asserted something false.
                "actual_locator": [q["retrieved"] for q in found],
                "why": "span is verbatim but on a page the answer did not cite" + amb}

    if drifted:
        return _dropped(claim, span, "drifted", cited, n, drifted[0][1], False)
    if undecided:
        return _undecidable(claim, span, cited, n, undecided[0][1], False)

    # 3) nowhere in the retrieved set
    return {"claim": claim["text"], "verdict": "unfounded", "span": span,
            "cited_page": cited, "citation_matched": n,
            "why": "span does not occur verbatim in any retrieved page" + amb}


def _as_ident(cited):
    """A citation may be a structured identity or a locator STRING.

    A model writes `manual.pdf#p.27`, not a dict, so accept it and parse it into the
    structured form the comparison uses. Parsing is what makes the string form real
    rather than decorative: an unparsed string compares equal to nothing and would
    silently never match.
    """
    if isinstance(cited, str):
        try:
            import locator
            return locator.parse(cited)
        except Exception:
            return {"doc": cited}
    return cited


def _same_page(retrieved, cited):
    """A page matches the citation when their identity fields agree. Loose by
    design: the appliance cites {id, path, page} and the Document's retrieved
    carries the same — extra fields (scores, url) never block a match."""
    if not isinstance(retrieved, dict) or not isinstance(cited, dict):
        return retrieved == cited
    keys = set(cited) & set(retrieved)
    return bool(keys) and all(retrieved.get(k) == cited.get(k) for k in keys)


def ground_answer(question, claims, pages):
    """-> a grounded answer or an abstention.

    {"question", "grounded": bool, "kept": [...grounded claims...],
     "dropped": [...misattributed/ambiguous/unfounded, with reasons...],
     "abstain": bool, "note": str}
    """
    verdicts = [verify_claim(c, pages) for c in claims]
    kept = [v for v in verdicts if v["verdict"] == "grounded"]
    dropped = [v for v in verdicts if v["verdict"] != "grounded"]
    if not kept:
        return {"question": question, "grounded": False, "abstain": True,
                "kept": [], "dropped": dropped,
                "note": "The library does not contain support for this — no claim "
                        "quoted a span verbatim from a cited page. Abstaining rather "
                        "than answer from retrieval alone."}
    return {"question": question, "grounded": True, "abstain": False,
            "kept": kept, "dropped": dropped,
            "note": "Grounded = each kept span is verbatim on the page cited. "
                    "Support of claim by span is NOT checked here (C3, needs the device)."}
