"""Re-runnable C3 characterization. The batch behind MEASUREMENT-c3.md.

    python3 c3_batch.py                 # list the fixtures, touch nothing
    CORDEXA_HOME=~/Projects/cordexa/main python3 c3_batch.py --run

THE ORIGINAL RUNNER WAS NOT VERSIONED. The 2026-09-05 n=6 table was produced by
an ad-hoc script that left `c3_batch_results.json` and `c3_batch.log` on
the workstation and nothing else — so the numbers in MEASUREMENT-c3.md could be read
but not reproduced. These fixtures are a RECONSTRUCTION from that table's case
names and descriptions, over page text that is really in this repo's fixtures.

Comparable in shape, not identical in text: G3 in the original was a radio-manual
passage whose text was not kept anywhere, so G3 here is a second verbatim case
over the drying pages. Do not report a re-run as the same six cases.

NPU CONCURRENCY — read before `--run`. C3 loads a 30B while the embedder may be
resident. Sustained concurrent NPU load is worth avoiding on this hardware: it
has cost this project a recovery that stopped BOTH models and took 48s, against
the 6s a start-only recovery takes. Run this when the embedder is idle, or stop
it deliberately for the duration. `--run` says so and pauses.
"""
import argparse
import json
import sys
import time
import urllib.request

import c3

# The management API, which names models properly. Measured 2026-09-05: a chat
# completion on this device reports `model: "../model/model.q8_0.gguf"` -- a
# generic path, identical whatever is loaded -- while the management API reports
# `Qwen/Qwen3-30B-A3B-Instruct`. So cordexa's served-string identity check, which
# compares what the completion returns between askings, is WEAK here: a swap
# between two chat models could report the same path both times and pass. This
# cross-check is the strong version for this device.
MGMT = "http://p8800.api.tiiny/api/v1/models/running"


def running_models():
    try:
        req = urllib.request.Request(
            MGMT, headers={"Authorization": "Bearer %s" % c3._token()})
        with urllib.request.urlopen(req, timeout=15) as r:
            return sorted(json.load(r).get("running") or [])
    except Exception as e:                      # noqa: BLE001 — advisory, never fatal
        return ["<unavailable: %s>" % e]

# Page text lifted from test_ground_answer.py — really from the harvested docs.
DRYING = ("Globally, drying is the most widely used method for preserving foods for use in "
          "the home or for sale. The most common method involves simply laying the product "
          "in the sun on mats, roofs or drying floors. This is known as sun drying.")
PRESERVE = ("Drying removes the moisture from the food so bacteria, yeast and mold cannot "
            "grow and spoil the food. It takes several days to dry foods out-of-doors.")

CASES = [
    {"name": "G1 verbatim-genuine", "expect": "supported",
     "span": "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
     "claim": "Drying removes moisture so that bacteria, yeast and mold cannot grow.",
     "why": "the claim restates the span with nothing added"},
    {"name": "G2 paraphrase-genuine", "expect": "supported",
     "span": "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
     "claim": "Drying prevents microbial growth by removing moisture.",
     "why": "a fair paraphrase — 'microbial growth' for the three organisms named. "
            "The original run HELD this one (orderings disagreed): the recall cost"},
    {"name": "G3 verbatim-drying", "expect": "supported",
     "span": "Globally, drying is the most widely used method for preserving foods",
     "claim": "Drying is the most widely used method of preserving food.",
     "why": "second verbatim case. NOT the original G3 (a radio manual, text not kept)"},
    {"name": "O1 overreach-ch2z_031", "expect": "not supported",
     "span": "Globally, drying is the most widely used method for preserving foods",
     "claim": "Sun drying is the most widely used method of preserving food.",
     "why": "THE motivating case: the claim adds 'sun', which the span does not state. "
            "C1 passes this — the span is verbatim — and only C3 can catch it"},
    {"name": "O2 specialization", "expect": "not supported",
     "span": "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
     "claim": "Drying removes moisture so that botulism spores cannot grow.",
     "why": "a specialization to one organism the span never names"},
    {"name": "R1 refuted-numeral", "expect": "not supported",
     "span": "It takes several days to dry foods out-of-doors",
     "claim": "It takes about thirty days to dry foods out-of-doors.",
     "why": "a number the span contradicts rather than omits"},
]


def run_one(case):
    t0 = time.time()
    r = c3.support_check(case["span"], case["claim"])
    r["seconds"] = round(time.time() - t0, 1)
    r["name"], r["expect"], r["why"] = case["name"], case["expect"], case["why"]
    if r["measured"]:
        got = "supported" if r["supported"] else "not supported"
        r["match_expected"] = (got == case["expect"])
    else:
        r["match_expected"] = None      # unmeasured is not a miss; it is no measurement
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="actually call the device")
    ap.add_argument("--out", default="c3_batch_results.json")
    a = ap.parse_args()

    if not a.run:
        for c in CASES:
            print(f"{c['name']:26} expect={c['expect']:14} {c['why']}")
        print(f"\n{len(CASES)} fixtures. --run to measure (reads the NPU warning above first).")
        return 0

    before = running_models()
    print(f"model {c3.MODEL} at {c3.DEVICE}")
    print(f"loaded before the run: {before}")
    print("NPU: this loads a 30B. If the embedder is live you are running both "
          "under concurrent NPU load.\n")
    results = []
    for c in CASES:
        r = run_one(c)
        results.append(r)
        if r["measured"]:
            got = "supported" if r["supported"] else "not supported"
            flag = "OK" if r["match_expected"] else "MISS"
            print(f"{r['name']:26} expect={r['expect']:14} A={str(r['a']):14} "
                  f"B={str(r['b']):14} -> {got:14} {flag}  [{r['served']}] {r['seconds']}s")
        else:
            print(f"{r['name']:26} UNMEASURED — {r['detail']}")

    measured = [r for r in results if r["measured"]]
    genuine = [r for r in measured if r["expect"] == "supported"]
    adverse = [r for r in measured if r["expect"] == "not supported"]
    print(f"\nmeasured {len(measured)}/{len(results)}"
          f"   false-accepts {sum(1 for r in adverse if r['supported'])}/{len(adverse)}"
          f"   recall {sum(1 for r in genuine if r['supported'])}/{len(genuine)}")

    # The identity fact the first run could not record, from both sides.
    served = sorted({str(r["served"]) for r in measured})
    after = running_models()
    print("served string across the run:", served or "(none measured)")
    if len(served) > 1:
        print("  ^ MORE THAN ONE. These askings were not judged by one model.")
    print(f"loaded after the run:  {after}")
    if before != after:
        print("  ^ THE LOADED MODEL SET CHANGED DURING THE RUN — the verdicts above "
              "were not all produced by the same model. Treat them as unmeasured.")

    traces = [r["trace"] for r in results if r.get("trace")]
    if traces:
        print(c3._c3().trace_flag_line(c3._c3().trace_flag_counts(traces)))

    with open(a.out, "w") as f:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                   "model_requested": c3.MODEL, "served": served,
                   "loaded_before": before, "loaded_after": after,
                   "fixtures": "reconstructed — see c3_batch.py docstring",
                   "results": results}, f, indent=1, default=str)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
