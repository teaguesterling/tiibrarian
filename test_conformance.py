"""Conformance against cordexa's golden set. Read-only — the set is not ours.

    CORDEXA_HOME=~/Projects/cordexa/main python3 -m unittest test_conformance -v

`examples/golden/README.md` is explicit: "Owner: the instrument owner. Nobody
else adds, edits, or removes a case." This module only READS it, and skips
cleanly when it is not on disk, so tiibrarian still tests standalone.

Why it exists: every other test here asserts *through* C1 on fixtures we wrote
ourselves, which cannot tell us whether this layer uses C1 the way the spec
means it to be used. The golden set can — its cases carry `targets` (which check
should decide) and `expect` (the star level the pipeline must land on), so it is
an outside judge rather than a mirror.

It earned its place immediately. On the first run (2026-09-05) it found that this
layer grounded `c09` — the resolution-drift case `spec/checks.md` C1.8 requires to
fail, and whose own note says it is "the failure that shipped 10 of 25 rows
elsewhere". See `_drift` in ground_answer.py.
"""
import json
import os
import unittest

import ground_answer as G

GOLDEN = os.path.join(G.CORDEXA, "examples", "golden")
HAVE = os.path.isdir(GOLDEN)


def _load():
    with open(os.path.join(GOLDEN, "documents.json")) as f:
        docs = {d["id"]: d for d in json.load(f)}
    with open(os.path.join(GOLDEN, "cases.json")) as f:
        cases = json.load(f)
    return docs, cases


def _run(case, doc):
    """This layer's verdict for one golden case, cited exactly as the case cites."""
    cites = case.get("cite", doc["retrieved"])
    return G.verify_claim({"text": case["claim"], "span": case["span"], "cites": cites},
                          [doc])


@unittest.skipUnless(HAVE, "cordexa golden set not present at %s" % GOLDEN)
class GoldenC1(unittest.TestCase):
    """The C1-target cases. `expect` is a star level for the WHOLE pipeline, so the
    testable statement here is about disposition, not label: a case cordexa expects
    at 0 must not be kept by this layer, and a case it expects at 2 must be."""

    @classmethod
    def setUpClass(cls):
        docs, cases = _load()
        cls.docs = docs
        cls.cases = [c for c in cases
                     if "C1" in (c.get("targets") or []) and c.get("doc")]

    def test_the_set_is_actually_loaded(self):
        # A conformance suite that silently tests nothing is worse than none.
        self.assertGreaterEqual(len(self.cases), 10)

    def test_zero_star_cases_are_never_kept(self):
        bad = []
        for c in self.cases:
            if c["expect"] != 0:
                continue
            v = _run(c, self.docs[c["doc"]])
            if v["verdict"] == "grounded":
                bad.append("%s (%s) -> grounded; case says: %s"
                           % (c["id"], c["why"][:60], c["expect"]))
        self.assertEqual(bad, [], "kept a claim cordexa expects at 0 stars:\n" + "\n".join(bad))

    def test_two_star_cases_are_kept(self):
        bad = []
        for c in self.cases:
            if c["expect"] != 2:
                continue
            v = _run(c, self.docs[c["doc"]])
            if v["verdict"] != "grounded":
                bad.append("%s -> %s (%s)" % (c["id"], v["verdict"], v["why"][:70]))
        self.assertEqual(bad, [], "dropped a claim cordexa expects at 2 stars:\n" + "\n".join(bad))

    def test_c1_8_resolution_drift_is_not_grounded(self):
        """The specific case this suite was written by, named so a regression says so.

        c09: 'Mint' requested, 'Spearmint' returned, span genuinely in the
        retrieved document. `spec/checks.md` C1.8 -> fail.
        """
        c = next((c for c in self.cases if c["id"] == "c09"), None)
        if c is None:
            self.skipTest("c09 not in this golden set")
        v = _run(c, self.docs[c["doc"]])
        self.assertEqual(v["verdict"], "drifted")
        self.assertIn("requested", v["why"])

    def test_drift_is_reported_as_drift_not_as_confabulation(self):
        # The span IS real. Calling it `unfounded` would blame the model for the
        # retriever's miss — the same error class as the abstain collapse.
        c = next((c for c in self.cases if c["id"] == "c09"), None)
        if c is None:
            self.skipTest("c09 not in this golden set")
        v = _run(c, self.docs[c["doc"]])
        self.assertNotIn(v["verdict"], ("unfounded", "grounded"))

    def test_wrong_citation_on_a_real_span_is_misattributed(self):
        """Where this layer is MORE specific than C1, and deliberately so.

        c08/f03/f06 cite a document the span is not in. C1 folds that into one
        `fail` ("cited locator is not the retrieved document"); this layer
        separates "you cited the wrong page" from "these words do not exist",
        because only the first is repairable. Both drop the claim, so this is a
        refinement of C1's verdict and not a disagreement with it.
        """
        for cid in ("c08", "f03", "f06"):
            c = next((c for c in self.cases if c["id"] == cid), None)
            if c is None:
                continue
            v = _run(c, self.docs[c["doc"]])
            self.assertEqual(v["verdict"], "misattributed", "%s -> %s" % (cid, v["verdict"]))
            self.assertNotEqual(v["verdict"], "grounded")


@unittest.skipUnless(HAVE, "cordexa golden set not present")
class WhatThisDoesNotProve(unittest.TestCase):
    """Stated as a test so the limit stays true rather than becoming a stale comment."""

    def test_c2_and_c3_cases_are_out_of_scope_here(self):
        _, cases = _load()
        out = [c["id"] for c in cases
               if not ({"C1"} & set(c.get("targets") or []))]
        self.assertTrue(out, "if every case targets C1, this module's scope note is wrong")
        # C2 (numeric/entity) and C3 (support) are not this module's job: C3 lives
        # in c3.py and needs the device; C2 is not implemented here at all. A green
        # run of this file is NOT a claim that the appliance conforms to checks.md.


if __name__ == "__main__":
    unittest.main()
