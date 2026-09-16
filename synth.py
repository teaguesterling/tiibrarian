"""Synthesis on the device: retrieved pages in, quoted claims out.

The last mock. `seam.mock_grounded` quotes the top hit so the loop can run without a
device; this asks the Tiiny's own LLM to answer, and it must answer in the shape the
grounding layer checks: every claim carries a SPAN it quotes and a LOCATOR it cites.

WHAT THIS DOES NOT DO: make the answer true. The model is asked to quote verbatim and
cite precisely, and it will sometimes do neither -- that is the entire reason
ground_answer exists downstream. A span that is not on the cited page is dropped whatever
the model asserted, so the prompt is a request, never a guarantee. Nothing here validates
spans; doing it in two places would let the two disagree.

THE MODEL IS NOT ASKED TO CITE. We know which pages were shown to it and we know what it
quoted, so where the quote came from is a fact `ground_answer.locate_span` establishes.
Asking the model for a value we can compute only adds a way to be wrong -- and it was a
real way: a mistyped locator is indistinguishable from a citation of a page that was
never retrieved, so a typo read as confabulation. Dropping the ask also shortens the
instruction, which leaves the model with one job: quote exactly.

A citation is still ACCEPTED if a model volunteers one (by index or as a locator string),
and it is then cross-checked rather than believed: a claim whose span sits somewhere other
than where it says is `misattributed`, and the verdict carries the true location. That
disagreement is worth surfacing rather than silently overwriting.
"""
import json
import os
import re
import urllib.request

import locator

CHAT_URL = os.environ.get("TIINY_CHAT_URL", "http://api.tiiny/v1/chat/completions")
AUTH = os.environ.get("TIINY_AUTH_KEY", "")
# Must be a model id the device actually has -- `Qwen/Qwen3-30B-A3B` does NOT exist;
# the real ids are ...-A3B-Instruct and ...-A3B-Thinking. Default is the small one
# so a first run costs the NPU least while an embedding job may be resident.
MODEL = os.environ.get("TIINY_CHAT_MODEL", "Qwen/Qwen3-8B")
# MEASURED against the device gateway, which cuts a request off at 60s. The 30B does
# ~69 tok/s of prompt processing and ~16 tok/s of generation, so the whole budget is
# roughly 3.5k prompt tokens AND a bounded answer. 5 pages x 2000 chars blew through it
# and returned HTTP 504 with nothing -- a timeout, not an error the model produced.
MAX_PAGE_CHARS = int(os.environ.get("TIIBRARIAN_MAX_PAGE_CHARS", "900"))
MAX_TOKENS = int(os.environ.get("TIIBRARIAN_MAX_TOKENS", "700"))

SYSTEM = (
    "You answer only from the pages given to you. For every claim you make, quote a span "
    "of text COPIED EXACTLY, character for character, from those pages. Do not paraphrase "
    "inside a span -- the span is checked against the pages automatically, and a claim "
    "whose span is not found word for word is discarded. Quote a WHOLE CLAUSE, not two "
    "or three words: very short spans are rejected as not identifying. You do NOT need to say which "
    "page a span came from; that is worked out from the span itself. If the pages do not "
    "answer the question, return no claims at all rather than guessing -- an empty answer "
    "is a correct answer when the library does not cover something.\n\n"
    'Reply with JSON only: {"claims": [{"text": "...", "span": "..."}]}'
)


def _locator_for(hit):
    """The locator string for a hit, whatever retrieval produced it."""
    if hit.get("locator"):
        return hit["locator"]
    if "page" in hit and hit.get("book"):
        return locator.render({"doc": hit["book"], "page": hit["page"]})
    if hit.get("source") and "n" in hit:          # seam.py's fts hits
        return str(hit["source"])
    return str(hit.get("book") or hit.get("source") or "")


def build_messages(question, hits, max_chars=MAX_PAGE_CHARS):
    """The prompt. Pure -- no device, so it is testable on its own."""
    pages = []
    for i, h in enumerate(hits, 1):
        text = (h.get("text") or h.get("content") or "")[:max_chars]
        pages.append(f"[{i}] {_locator_for(h)}\n{text}")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": "PAGES:\n\n" + "\n\n---\n\n".join(pages) +
                        f"\n\nQUESTION: {question}"}]


def _extract_json(raw):
    """The model's JSON, out of whatever it wrapped it in.

    Tolerant on purpose: a fenced block or a sentence of preamble is a formatting slip,
    not a wrong answer, and rejecting it would throw away a good answer over punctuation.
    """
    raw = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(raw[i:j + 1])
        except Exception:
            return None
    return None


def parse_claims(raw, hits):
    """Model output -> claims ground_answer can check. Unusable claims are DROPPED.

    Dropped, not repaired: a claim with no span cannot be grounded, and inventing one
    would be this layer asserting something the model did not say.
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return []
    out = []
    for c in data.get("claims") or []:
        if not isinstance(c, dict):
            continue
        span = (c.get("span") or "").strip()
        if not span:
            continue
        cite, ref = c.get("cite"), None
        if isinstance(cite, bool):
            ref = None
        elif isinstance(cite, int) and 1 <= cite <= len(hits):
            ref = _locator_for(hits[cite - 1])
        elif isinstance(cite, str):
            s = cite.strip()
            if s.isdigit() and 1 <= int(s) <= len(hits):
                ref = _locator_for(hits[int(s) - 1])
            elif s:
                ref = s                      # the model wrote a locator; take it as given
        # ref may be None, and that is the NORMAL case now: no citation means the
        # locator is derived from where the span actually is. Only a citation that was
        # offered and is unusable (an out-of-range index) is discarded, because that is
        # the model asserting something false rather than declining to assert.
        if cite is not None and ref is None:
            continue
        out.append({"text": (c.get("text") or "").strip() or span,
                    "span": span, "cites": ref})
    return out


def device_synth(question, hits, timeout=120, temperature=0.0):
    """Ask the device's LLM. Raises without a token rather than degrading silently."""
    if not AUTH:
        raise RuntimeError(
            "synth.device_synth needs TIINY_AUTH_KEY and the device reachable at "
            f"{CHAT_URL}. Use seam.mock_grounded for a device-free run.")
    body = json.dumps({"model": MODEL, "messages": build_messages(question, hits),
                       "temperature": temperature,
                       # Bounded, or generation can outlast the gateway and the whole
                       # answer is lost to a 504 rather than returned short.
                       "max_tokens": MAX_TOKENS}).encode()
    req = urllib.request.Request(
        CHAT_URL, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + AUTH})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return parse_claims(payload["choices"][0]["message"]["content"], hits)
