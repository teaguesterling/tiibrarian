"""The C1 surface tiibrarian depends on — asserted in ONE place, by name.

    CORDEXA_HOME=~/Projects/cordexa/main python3 -m unittest test_c1_surface -v

Why this file exists, plainly: `ground_answer.py` imports `reference/c1_span.py`
from cordexa **by filesystem path**, and that file says of itself that it is a
reference implementation, not the normative spec. So tiibrarian is coupled to
behaviour that is allowed to change, with no version pin and no import error to
warn us — a replacement C1 would silently change verdicts on real answers and
every tiibrarian test would still pass, because they all assert *through* C1.

These tests assert the coupling itself. Each one names a property tiibrarian's
logic actually relies on and would be WRONG without. If cordexa changes one, a
test here fails with a message saying which assumption broke, instead of the
breakage arriving as quietly mislabelled answers.

They are deliberately about the CONTRACT, not the tuning. `MIN_SIGNIFICANT_CHARS`
is read from the module rather than written as `16`: that a floor exists is the
contract; where it sits is cordexa's to move.
"""
import unittest

import ground_answer as G

C1 = G._c1()

TEXT = ("Drying removes the moisture from the food so bacteria, yeast and mold cannot "
        "grow and spoil the food. It takes several days to dry foods out-of-doors.")
SPAN = "Drying removes the moisture from the food so bacteria"
IDENT = {"kind": "zim", "path": "preserving-food-drying.pdf", "page": 1}


def doc(text=TEXT, requested=None, retrieved=None, **extra):
    d = {"requested": requested if requested is not None else dict(IDENT),
         "retrieved": retrieved if retrieved is not None else dict(IDENT),
         "resolution": "exact", "text": text}
    d.update(extra)
    return d


class Signature(unittest.TestCase):
    """`check(span, document, cited_locator=...) -> (verdict, reason)`."""

    def test_returns_verdict_and_reason(self):
        got = C1.check(SPAN, doc(), cited_locator=dict(IDENT))
        self.assertIsInstance(got, tuple)
        self.assertEqual(len(got), 2, "ground_answer unpacks (verdict, reason)")
        self.assertIsInstance(got[0], str)
        self.assertIsInstance(got[1], str)

    def test_cited_locator_is_a_keyword(self):
        # _c1_verdicts passes it by name on every call.
        self.assertIn("cited_locator", C1.check.__code__.co_varnames)


class ThreeVerdicts(unittest.TestCase):
    """pass / fail / **abstain**. All three, and abstain is not a soft fail.

    This is the property that was already violated once: ground_answer collapsed
    abstain into "not found" and reported `unfounded` — an accusation of
    confabulation — for a span C1 had explicitly declined to judge.
    """

    def test_pass_is_reachable(self):
        v, _ = C1.check(SPAN, doc(), cited_locator=dict(IDENT))
        self.assertEqual(v, "pass")

    def test_fail_is_reachable(self):
        v, why = C1.check("this sentence is nowhere in the retrieved page at all",
                          doc(), cited_locator=dict(IDENT))
        self.assertEqual(v, "fail")
        self.assertTrue(why)

    def test_abstain_is_reachable_and_distinct(self):
        # Non-Latin letters the fold cannot compare. Real for this corpus:
        # chemistry, medicine and mathematics pages carry Greek constantly.
        text = "The Greek letter alpha, written άλφα, denotes the first quantity."
        v, why = C1.check("Greek letter alpha, written άλφα, denotes the first",
                          doc(text), cited_locator=dict(IDENT))
        self.assertEqual(v, "abstain")
        self.assertNotEqual(v, "fail")
        self.assertTrue(why, "ground_answer surfaces this reason to the reader")

    def test_vocabulary_is_closed(self):
        # Every verdict ground_answer can see must be one it branches on.
        seen = set()
        for span, d in [(SPAN, doc()),
                        ("this sentence is nowhere in the retrieved page", doc()),
                        ("Greek letter alpha, written άλφα, denotes the first",
                         doc("The Greek letter alpha, written άλφα, denotes the first.")),
                        (SPAN, doc(retrieved={"kind": "zim", "path": "other.pdf", "page": 9})),
                        ("too short", doc())]:
            seen.add(C1.check(span, d, cited_locator=d["requested"])[0])
        self.assertTrue(seen <= {"pass", "fail", "abstain"},
                        "unknown verdict %s — ground_answer would treat it as 'not pass'" % seen)


class TextIsReadFromTheDocument(unittest.TestCase):
    """The document decides, not the span.

    `locate_span` is tiibrarian's whole attribution story: it asks C1 the same
    span against every retrieved page and calls the passes "where the quote came
    from". That is only true if C1 actually reads each document's `text`.
    """

    def test_same_span_different_text_different_verdict(self):
        a, _ = C1.check(SPAN, doc(TEXT), cited_locator=dict(IDENT))
        b, _ = C1.check(SPAN, doc("Entirely unrelated prose about bicycle repair."),
                        cited_locator=dict(IDENT))
        self.assertEqual(a, "pass")
        self.assertEqual(b, "fail")

    def test_missing_text_is_a_failure_not_a_pass(self):
        d = doc()
        del d["text"]
        self.assertEqual(C1.check(SPAN, d, cited_locator=dict(IDENT))[0], "fail")

    def test_unavailable_document_is_a_failure_not_a_pass(self):
        v, _ = C1.check(SPAN, doc(unavailable="fetch timed out"), cited_locator=dict(IDENT))
        self.assertEqual(v, "fail")


class LocatorRule(unittest.TestCase):
    """`requested ⊆ claimed`, and every claimed key equals the RETRIEVED value.

    tiibrarian leans on this twice: `_c1_verdicts` passes `p["retrieved"]` as the
    cited locator, and `verify_claim` treats a pass as proof the span is on *that*
    page. If C1 stopped comparing against `retrieved`, `misattributed` — the
    defect this layer exists to catch — would stop being detectable.
    """

    def test_retrieved_as_cited_locator_matches(self):
        self.assertEqual(C1.check(SPAN, doc(), cited_locator=dict(IDENT))[0], "pass")

    def test_citation_vaguer_than_the_request_does_not_match(self):
        # Dropping `page` must not match by omission — that is how two printings
        # of one manual would silently collapse into each other.
        v, why = C1.check(SPAN, doc(), cited_locator={"kind": "zim",
                                                      "path": "preserving-food-drying.pdf"})
        self.assertEqual(v, "fail")
        self.assertIn("locator", why)

    def test_wrong_value_does_not_match(self):
        v, _ = C1.check(SPAN, doc(), cited_locator=dict(IDENT, page=9))
        self.assertEqual(v, "fail")

    def test_extra_fields_on_retrieved_do_not_block(self):
        # A real adapter's `retrieved` carries url/score/title the request never had.
        d = doc(retrieved=dict(IDENT, url="file:///x.pdf", score=0.81))
        self.assertEqual(C1.check(SPAN, d, cited_locator=dict(IDENT))[0], "pass")

    def test_compared_against_retrieved_not_requested(self):
        # requested and retrieved disagree; citing the REQUESTED page must not pass.
        d = doc(requested=dict(IDENT, page=1), retrieved=dict(IDENT, page=2),
                resolution="nearest")
        self.assertEqual(C1.check(SPAN, d, cited_locator=dict(IDENT, page=1))[0], "fail")


class WhatPassMeans(unittest.TestCase):
    """"Verbatim" means whole tokens, and there is a length floor."""

    def test_mid_token_match_is_not_verbatim(self):
        # tiibrarian reports a pass to the user as "these words are on that page".
        v, _ = C1.check("oisture from the foo", doc(), cited_locator=dict(IDENT))
        self.assertNotEqual(v, "pass")

    def test_a_length_floor_exists(self):
        floor = getattr(C1, "MIN_SIGNIFICANT_CHARS", None)
        self.assertIsInstance(floor, int, "synth's span-length guidance assumes a floor")
        # Derived from the declared value, so cordexa may move it without a
        # false failure here. What is asserted is that the floor BINDS.
        short = C1.norm(TEXT)[:max(0, floor - 1)]
        self.assertLess(len(short), floor)
        v, _ = C1.check(short, doc(), cited_locator=dict(IDENT))
        self.assertNotEqual(v, "pass", "a sub-floor span must not ground a claim")

    def test_whitespace_differences_do_not_matter(self):
        # The loosening we asked cordexa for: verbatim AFTER whitespace normalisation,
        # so a span pulled across a reader's line wrap still grounds.
        v, _ = C1.check("Drying   removes\n the moisture\tfrom the food", doc(),
                        cited_locator=dict(IDENT))
        self.assertEqual(v, "pass")


if __name__ == "__main__":
    unittest.main()
