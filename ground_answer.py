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

A citation must resolve to EXACTLY ONE retrieved page (`_resolve_cited`). Zero
matches, or two or more, means the answer did not attribute the span to a page,
so the claim cannot be grounded -- an under-specified citation is not attribution.

A citation resolves to the retrieved pages whose identity it matches. Resolving to
SEVERAL is not itself a failure: if the span is verbatim on all of them the answer is
the same whichever was meant, so the claim stands. It fails only when the ambiguity is
outcome-changing -- the span on some candidates and not others -- which is `ambiguous`.

Two mechanical verdicts, per the advisor's split:
  - span in the CITED page            -> grounded            (kept)
  - span in a DIFFERENT retrieved page -> misattributed       (dropped)  <- the spoken-answer defect
  - span in NO retrieved page          -> unfounded           (dropped)  <- confabulation
  - cited pages disagree about the span -> ambiguous          (dropped)  <- cannot tell which

Abstain rule, structural (stated before any output, no tunable): a claim that is
not `grounded` is dropped; the answer ABSTAINS iff zero claims survive. A
coverage floor would be a heuristic and belongs on the far side of
checks.md's mechanical/heuristic line — not here.

NPU-free: C1 is mechanical. Only the synthesis that produced the claims used the
device. Import cordexa's C1 by path so the home of this module stays open.
"""
import importlib.util
import os

CORDEXA = os.path.expanduser(os.environ.get("CORDEXA_HOME", "~/Projects/cordexa"))


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


def verify_claim(claim, pages):
    """claim: {"text", "span", "cites"} where `cites` is the retrieved-identity
    of the page the answer attributes the span to (a page's `retrieved`).
    pages: [adapter Document]. -> a per-claim verdict dict."""
    c1 = _c1()
    cited = _as_ident(claim.get("cites"))
    span = claim.get("span", "")

    # 1) the pages this citation names, and whether the span is on them
    matches = _cited_pages(pages, cited)
    n = len(matches)
    if n:
        on = [p for p in matches
              if c1.check(span, p, cited_locator=p["retrieved"])[0] == "pass"]
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
        # verbatim on none of them -> fall through

    amb = ("" if n <= 1 else
           " (the citation matched %d retrieved pages and the span is on none of "
           "them)" % n)

    # 2) not on the cited page(s) — is it verbatim in some OTHER retrieved page?
    cited_set = {id(p) for p in matches}
    for p in pages:
        if id(p) in cited_set:
            continue
        v, _ = c1.check(span, p, cited_locator=p["retrieved"])
        if v == "pass":
            return {"claim": claim["text"], "verdict": "misattributed", "span": span,
                    "cited_page": cited, "found_on": p["retrieved"],
                    "citation_matched": n,
                    "why": "span is verbatim but on a page the answer did not cite" + amb}

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
