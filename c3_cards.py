"""Run C3 over a batch of grimoire cards. The handoff half of the agreed seam.

    python3 c3_cards.py c3-requests.json -o c3-verdicts.json

IN   {"cards":[{card_id, claim, span, content_sha256}, ...]}   (a bare list is fine too)
OUT  {"cards":[{card_id, content_sha256, verdict, note}, ...]}

`verdict` is one of supported | not_supported | unmeasured.

WHY THIS SIDE OWNS THE DEVICE CALL. grimoire could load `c3.py` by path and
inject its own `ask`, and that would quietly lose four guards that are specific
to this appliance: the served-model identity cross-check (a chat completion here
reports a GENERIC `../model/model.q8_0.gguf` whatever is loaded, so cordexa's
own identity test is weak and `c3_batch` compensates against the management
API); the ~60s bodiless-504 gateway budget; retry-transport-only (a reply that
ARRIVED is never re-asked); and the unparseable-but-complete -> unmeasured
override. cordexa's `check` also merges its trace metadata only when
`ask is call_device`, so an injected ask drops it.

UNMEASURED IS NOT A NEGATIVE VERDICT. It means no verdict was reached -- device
down, gateway cut, no ANSWER line, or the served model changed between askings.
The consumer records it and leaves the card where it was.

CONTENT_SHA256 IS ECHOED, NEVER CHECKED. We do not resolve it; it is a token
grimoire issued and verifies on ingest. Echoing it unexamined is the point: this
side must not be able to assert that a span is the one grimoire grounded on.
"""
import argparse
import json
import sys
import time

import c3


def verdict_for(card, check=None):
    """-> {card_id, content_sha256, verdict, note}. Never raises for one card."""
    check = check or c3.support_check
    out = {"card_id": card.get("card_id"),
           "content_sha256": card.get("content_sha256")}
    span, claim = card.get("span"), card.get("claim")
    if not span or not claim:
        out.update(verdict="unmeasured", note="missing span or claim in the request")
        return out
    try:
        s = check(span, claim)
    except Exception as e:                       # noqa: BLE001 — one card must not kill a batch
        out.update(verdict="unmeasured", note="%s: %s" % (type(e).__name__, str(e)[:200]))
        return out
    if not s.get("measured"):
        out.update(verdict="unmeasured", note=str(s.get("detail") or "no verdict")[:300])
    else:
        out.update(verdict="supported" if s["supported"] else "not_supported",
                   note=str(s.get("detail") or "")[:300])
    return out


def run(cards, check=None, log=lambda *a: None):
    res = []
    for i, card in enumerate(cards, 1):
        t0 = time.time()
        v = verdict_for(card, check)
        log("  [%d/%d] %-28s %-14s %.1fs" % (i, len(cards), str(v["card_id"])[:28],
                                             v["verdict"], time.time() - t0))
        res.append(v)
    return res


def _cards(blob):
    return blob["cards"] if isinstance(blob, dict) else blob


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("-o", "--out", default="c3-verdicts.json")
    a = ap.parse_args()

    cards = _cards(json.load(open(a.infile)))
    print("%d card(s) from %s — model %s at %s" % (len(cards), a.infile, c3.MODEL, c3.DEVICE))
    if not cards:
        print("nothing to do")
    res = run(cards, log=lambda m: print(m, flush=True))

    n = {}
    for r in res:
        n[r["verdict"]] = n.get(r["verdict"], 0) + 1
    print("\n" + "  ".join("%s %d" % kv for kv in sorted(n.items())))
    if n.get("unmeasured") == len(res) and res:
        print("EVERY card unmeasured — read this as a device condition, not 200 nulls.")
    with open(a.out, "w") as f:
        json.dump({"cards": res}, f, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
