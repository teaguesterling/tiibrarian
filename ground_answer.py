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

Two mechanical verdicts, per the advisor's split:
  - span in the CITED page            -> grounded            (kept)
  - span in a DIFFERENT retrieved page -> misattributed       (dropped)  <- the spoken-answer defect
  - span in NO retrieved page          -> unfounded           (dropped)  <- confabulation

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


def _resolve_cited(pages, cited):
    """The single retrieved page a claim cites -> (page|None, n_matched).

    A citation must identify EXACTLY ONE retrieved page. Zero matches means the
    answer cited nothing that was retrieved. Two or more means the citation is
    under-specified: it names a *family* of pages, not a page, and an
    under-specified citation is not attribution. Either way there is no cited
    page, so the claim cannot be `grounded`.

    Without this, a citation carrying only generic fields (e.g. {kind, id} but
    no path/page) agrees with every page, so `_same_page` matches them all and
    the claim silently binds to whichever page happened to be retrieved first --
    handing a free pass to exactly the misattribution this layer exists to
    catch. An LLM emitting a partial citation is the expected case, not an
    exotic one.
    """
    matches = [p for p in pages if _same_page(p.get("retrieved"), cited)]
    return (matches[0] if len(matches) == 1 else None), len(matches)


def verify_claim(claim, pages):
    """claim: {"text", "span", "cites"} where `cites` is the retrieved-identity
    of the page the answer attributes the span to (a page's `retrieved`).
    pages: [adapter Document]. -> a per-claim verdict dict."""
    c1 = _c1()
    cited = claim.get("cites")
    span = claim.get("span", "")

    # 1) is the span in the page the claim CITES?
    cited_page, n_matched = _resolve_cited(pages, cited)
    if cited_page is not None:
        v, why = c1.check(span, cited_page, cited_locator=cited_page["retrieved"])
        if v == "pass":
            return {"claim": claim["text"], "verdict": "grounded", "span": span,
                    "cited_page": cited, "citation_matched": 1,
                    "support_checked": False, "why": ""}

    amb = ("" if n_matched <= 1 else
           " (the citation matched %d retrieved pages — under-specified, so it "
           "identifies no single page)" % n_matched)

    # 2) not in the cited page — is it verbatim in some OTHER retrieved page?
    for p in pages:
        if p is cited_page:
            continue
        v, _ = c1.check(span, p, cited_locator=p["retrieved"])
        if v == "pass":
            return {"claim": claim["text"], "verdict": "misattributed", "span": span,
                    "cited_page": cited, "found_on": p["retrieved"],
                    "citation_matched": n_matched,
                    "why": "span is verbatim but on a page the answer did not cite" + amb}

    # 3) nowhere in the retrieved set
    return {"claim": claim["text"], "verdict": "unfounded", "span": span,
            "cited_page": cited, "citation_matched": n_matched,
            "why": "span does not occur verbatim in any retrieved page" + amb}


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
     "dropped": [...misattributed/unfounded, with reasons...],
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
