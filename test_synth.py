"""Prompt building and response parsing for device synthesis. Hermetic, no device."""
import unittest

import ground_answer as G
import synth

HITS = [
    {"n": 1, "source": "text", "book": "canning.pdf", "page": 27,
     "text": "Blanch the beans for three minutes before packing them into jars."},
    {"n": 2, "source": "text", "book": "drying.pdf", "page": 4,
     "text": "Sun drying requires several consecutive days of dry weather."},
]
PAGES = [{"requested": {"kind": "corpus", "doc": h["book"], "page": h["page"]},
          "retrieved": {"kind": "corpus", "doc": h["book"], "page": h["page"]},
          "resolution": "exact", "text": h["text"]} for h in HITS]


class Prompt(unittest.TestCase):
    def test_pages_are_numbered_and_carry_their_locator(self):
        m = synth.build_messages("how long to blanch?", HITS)
        user = m[1]["content"]
        self.assertIn("[1] canning.pdf#p.27", user)
        self.assertIn("[2] drying.pdf#p.4", user)
        self.assertIn("QUESTION: how long to blanch?", user)

    def test_pages_are_truncated_so_a_long_page_cannot_crowd_out_the_rest(self):
        big = [{"n": 1, "book": "b.pdf", "page": 1, "text": "x" * 10000}]
        user = synth.build_messages("q", big, max_chars=50)[1]["content"]
        self.assertLess(len(user), 500)

    def test_the_system_prompt_asks_for_abstention_not_a_guess(self):
        self.assertIn("rather than guessing", synth.SYSTEM)


class ParseClaims(unittest.TestCase):
    def test_index_citation_becomes_that_hits_locator(self):
        raw = '{"claims":[{"text":"Blanch three minutes.","span":"Blanch the beans for three minutes","cite":1}]}'
        c = synth.parse_claims(raw, HITS)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0]["cites"], "canning.pdf#p.27")

    def test_fenced_json_is_accepted(self):
        raw = 'Sure!\n```json\n{"claims":[{"text":"t","span":"Sun drying requires","cite":2}]}\n```'
        self.assertEqual(synth.parse_claims(raw, HITS)[0]["cites"], "drying.pdf#p.4")

    def test_a_locator_string_from_the_model_is_taken_as_given(self):
        raw = '{"claims":[{"text":"t","span":"Blanch the beans","cite":"canning.pdf#p.27"}]}'
        self.assertEqual(synth.parse_claims(raw, HITS)[0]["cites"], "canning.pdf#p.27")

    def test_claims_with_no_span_are_dropped_not_repaired(self):
        # A claim with no span cannot be grounded, and inventing one would make this
        # layer assert something the model did not say.
        raw = '{"claims":[{"text":"unsupported assertion","cite":1}]}'
        self.assertEqual(synth.parse_claims(raw, HITS), [])

    def test_an_out_of_range_index_is_dropped(self):
        # Citing page 9 of 2 is not a citation of anything retrieved.
        raw = '{"claims":[{"text":"t","span":"Blanch the beans","cite":9}]}'
        self.assertEqual(synth.parse_claims(raw, HITS), [])

    def test_no_claims_means_no_claims(self):
        # The abstention path: the model saying "the pages do not answer this" must
        # survive as an empty list, which ground_answer turns into an abstention.
        self.assertEqual(synth.parse_claims('{"claims":[]}', HITS), [])

    def test_garbage_does_not_raise(self):
        for raw in ("", None, "I don't know.", "{", '{"claims": "not a list"}'):
            self.assertEqual(synth.parse_claims(raw, HITS), [])

    def test_a_boolean_cite_is_not_an_index(self):
        # bool is an int in Python; True must not silently become hit [1].
        raw = '{"claims":[{"text":"t","span":"Blanch the beans","cite":true}]}'
        self.assertEqual(synth.parse_claims(raw, HITS), [])


class DerivedAttribution(unittest.TestCase):
    """The model quotes; we work out where the quote came from."""

    def test_a_claim_with_no_citation_is_kept(self):
        # This is the NORMAL shape now, not a degraded one.
        raw = '{"claims":[{"text":"Blanch three minutes.","span":"Blanch the beans for three minutes"}]}'
        c = synth.parse_claims(raw, HITS)
        self.assertEqual(len(c), 1)
        self.assertIsNone(c[0]["cites"])

    def test_grounding_derives_the_locator_from_the_span(self):
        raw = '{"claims":[{"text":"t","span":"Sun drying requires several"}]}'
        r = G.ground_answer("q", synth.parse_claims(raw, HITS), PAGES)
        self.assertTrue(r["grounded"])
        kept = r["kept"][0]
        self.assertTrue(kept["derived"])
        self.assertEqual(kept["cited_page"]["doc"], "drying.pdf")   # the RIGHT page
        self.assertEqual(kept["cited_page"]["page"], 4)

    def test_a_derived_claim_cannot_be_misattributed(self):
        # The point of deriving: every span in the retrieved set attributes correctly,
        # so misattribution is not merely unlikely, it is unreachable on this path.
        for span, doc in (("Blanch the beans for three minutes", "canning.pdf"),
                          ("Sun drying requires several", "drying.pdf")):
            r = G.ground_answer("q", [{"text": "t", "span": span, "cites": None}], PAGES)
            self.assertTrue(r["grounded"], span)
            self.assertEqual(r["kept"][0]["cited_page"]["doc"], doc)

    def test_an_uncited_confabulation_is_still_unfounded(self):
        # Deriving attribution must not become a way to launder invented text.
        r = G.ground_answer("q", [{"text": "t", "span": "boil for one hour",
                                   "cites": None}], PAGES)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["dropped"][0]["verdict"], "unfounded")
        self.assertTrue(r["dropped"][0]["derived"])

    def test_a_span_on_two_pages_derives_both(self):
        twin = dict(PAGES[0])
        twin["requested"] = twin["retrieved"] = {"kind": "corpus", "doc": "reprint.pdf",
                                                 "page": 27}
        r = G.ground_answer("q", [{"text": "t", "span": "Blanch the beans for three",
                                   "cites": None}], [PAGES[0], twin])
        self.assertTrue(r["grounded"])
        self.assertEqual(r["kept"][0]["citation_matched"], 2)
        self.assertEqual(len(r["kept"][0]["found_on"]), 2)

    def test_a_volunteered_wrong_citation_still_fails_and_records_the_truth(self):
        # A model that DOES cite is cross-checked, not believed -- and the verdict
        # carries the correct locator, so a caller can repair rather than guess.
        r = G.ground_answer("q", [{"text": "t", "span": "Sun drying requires several",
                                   "cites": "canning.pdf#p.27"}], PAGES)
        self.assertTrue(r["abstain"])
        d = r["dropped"][0]
        self.assertEqual(d["verdict"], "misattributed")
        self.assertEqual(d["actual_locator"][0]["doc"], "drying.pdf")

    def test_an_offered_but_unusable_citation_is_still_discarded(self):
        # Declining to cite is fine; citing page 9 of 2 is an assertion that is false.
        raw = '{"claims":[{"text":"t","span":"Blanch the beans","cite":9}]}'
        self.assertEqual(synth.parse_claims(raw, HITS), [])


class SpanLengthFloor(unittest.TestCase):
    def test_c1_rejects_spans_shorter_than_four_words(self):
        # MEASURED, and it bites: a three-word quote is `unfounded` even when it is
        # verbatim on the page. A synth that quotes tersely would have every claim
        # rejected with no hint why, so the prompt asks for a full clause and this
        # records the floor that makes that necessary.
        page = [{"requested": {"kind": "c", "doc": "a.pdf", "page": 1},
                 "retrieved": {"kind": "c", "doc": "a.pdf", "page": 1},
                 "resolution": "exact",
                 "text": "Blanch the beans for three minutes before packing them."}]
        short = G.ground_answer("q", [{"text": "t", "span": "Blanch the beans",
                                       "cites": None}], page)
        self.assertTrue(short["abstain"])
        self.assertEqual(short["dropped"][0]["verdict"], "unfounded")
        longer = G.ground_answer("q", [{"text": "t", "span": "Blanch the beans for",
                                        "cites": None}], page)
        self.assertTrue(longer["grounded"])


class EndToEnd(unittest.TestCase):
    """Parsed claims must flow into grounding unchanged."""

    def test_a_faithful_answer_grounds(self):
        raw = '{"claims":[{"text":"Blanch three minutes.","span":"Blanch the beans for three minutes","cite":1}]}'
        r = G.ground_answer("q", synth.parse_claims(raw, HITS), PAGES)
        self.assertTrue(r["grounded"])

    def test_a_confabulated_span_is_caught_downstream_not_here(self):
        # synth does NOT validate spans; doing it in two places would let the two
        # disagree. The model's invention reaches ground_answer and is dropped there.
        raw = '{"claims":[{"text":"t","span":"boil the beans for one hour","cite":1}]}'
        claims = synth.parse_claims(raw, HITS)
        self.assertEqual(len(claims), 1)              # synth passed it through
        r = G.ground_answer("q", claims, PAGES)
        self.assertTrue(r["abstain"])                 # grounding rejected it
        self.assertEqual(r["dropped"][0]["verdict"], "unfounded")

    def test_a_span_quoted_from_the_wrong_page_is_misattributed(self):
        raw = '{"claims":[{"text":"t","span":"Sun drying requires several","cite":1}]}'
        r = G.ground_answer("q", synth.parse_claims(raw, HITS), PAGES)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["dropped"][0]["verdict"], "misattributed")


class Guards(unittest.TestCase):
    def test_device_synth_without_a_token_raises(self):
        saved = synth.AUTH
        synth.AUTH = ""
        try:
            with self.assertRaises(RuntimeError) as e:
                synth.device_synth("q", HITS)
            self.assertIn("TIINY_AUTH_KEY", str(e.exception))
        finally:
            synth.AUTH = saved


if __name__ == "__main__":
    unittest.main()
