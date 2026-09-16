"""Answer-grounding fixtures + hermetic test. C1 is mechanical, so no device.

    CORDEXA_HOME=~/Projects/cordexa python3 -m unittest test_ground_answer -v

Real SurvivorLibrary text so the spans are genuinely verbatim; the two cases
that bite are misattribution (right words, wrong page) and the ch2z_031
over-reach (right words, right page, claim over-reaches — the C1/C3 boundary).
"""
import unittest

import ground_answer as G

# two real pages (verbatim from the harvested docs)
PAGE_SMALL = {
    "requested": {"kind": "zim", "id": "survivorlibrary", "path": "smallscale-food-drying-technologies.pdf", "page": 1},
    "retrieved": {"kind": "zim", "id": "survivorlibrary", "path": "smallscale-food-drying-technologies.pdf", "page": 1},
    "resolution": "exact",
    "text": ("Globally, drying is the most widely used method for preserving foods for use in "
             "the home or for sale. The most common method involves simply laying the product "
             "in the sun on mats, roofs or drying floors. This is known as sun drying."),
}
PAGE_PRESERVE = {
    "requested": {"kind": "zim", "id": "survivorlibrary", "path": "preserving-food-drying.pdf", "page": 1},
    "retrieved": {"kind": "zim", "id": "survivorlibrary", "path": "preserving-food-drying.pdf", "page": 1},
    "resolution": "exact",
    "text": ("Drying removes the moisture from the food so bacteria, yeast and mold cannot "
             "grow and spoil the food. It takes several days to dry foods out-of-doors."),
}
# Non-Latin text, which this corpus has in quantity: chemistry, medicine and
# mathematics pages carry Greek constantly. C1 ABSTAINS on these rather than
# guessing, and the layer must report that as its own thing.
PAGE_GREEK = {
    "requested": {"kind": "zim", "id": "survivorlibrary", "path": "practical-chemistry.pdf", "page": 12},
    "retrieved": {"kind": "zim", "id": "survivorlibrary", "path": "practical-chemistry.pdf", "page": 12},
    "resolution": "exact",
    "text": ("The Greek letter alpha, written άλφα, denotes the first quantity in the "
             "series, and the coefficient is read off the second column."),
}
PAGES = [PAGE_SMALL, PAGE_PRESERVE]


def claim(text, span, cites):
    return {"text": text, "span": span, "cites": cites}


class Grounding(unittest.TestCase):
    def test_grounded_claim_kept(self):
        c = claim("Drying prevents microbial growth by removing moisture.",
                  "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
                  PAGE_PRESERVE["retrieved"])
        r = G.ground_answer("How does drying preserve food?", [c], PAGES)
        self.assertTrue(r["grounded"])
        self.assertEqual(len(r["kept"]), 1)
        self.assertFalse(r["kept"][0]["support_checked"])   # grounded != supported

    def test_unfounded_claim_abstains(self):
        c = claim("Freezing is the best way to preserve all foods.",
                  "Freezing is the single best method for preserving every kind of food",
                  PAGE_PRESERVE["retrieved"])
        r = G.ground_answer("What is the best preservation method?", [c], PAGES)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["dropped"][0]["verdict"], "unfounded")

    def test_misattributed_right_words_wrong_page(self):
        # span is verbatim — but on PAGE_SMALL, while the claim cites PAGE_PRESERVE
        c = claim("Drying is the most widely used preservation method.",
                  "Globally, drying is the most widely used method for preserving foods",
                  PAGE_PRESERVE["retrieved"])
        r = G.ground_answer("Is drying widely used?", [c], PAGES)
        self.assertTrue(r["abstain"])                       # miscited -> dropped -> zero survive
        self.assertEqual(r["dropped"][0]["verdict"], "misattributed")
        self.assertEqual(r["dropped"][0]["found_on"]["path"], "smallscale-food-drying-technologies.pdf")

    def test_overreach_is_grounded_but_support_unchecked(self):
        # ch2z_031: verbatim span, CORRECT page, but the claim narrows drying -> sun drying.
        # C1 cannot see the over-reach; the layer keeps it and flags support unchecked.
        c = claim("Sun drying is the most widely used method for preserving foods globally.",
                  "Globally, drying is the most widely used method for preserving foods",
                  PAGE_SMALL["retrieved"])
        r = G.ground_answer("What is the most widely used method?", [c], PAGES)
        self.assertTrue(r["grounded"])
        self.assertEqual(r["kept"][0]["verdict"], "grounded")
        self.assertFalse(r["kept"][0]["support_checked"])   # the C3 boundary, documented

    def test_all_confabulated_answer_abstains(self):
        cs = [claim("Freezing beats everything.", "Freezing is universally superior for all foods", PAGE_PRESERVE["retrieved"]),
              claim("Salt is unnecessary.", "Salt plays no role whatsoever in any food preservation", PAGE_SMALL["retrieved"])]
        r = G.ground_answer("q", cs, PAGES)
        self.assertTrue(r["abstain"])
        self.assertEqual(len(r["kept"]), 0)

    # --- citation resolution: an under-specified citation is not attribution ---

    def test_ambiguity_that_changes_the_verdict_is_refused(self):
        # The span is verbatim on PAGE_SMALL only, but the citation carries just the
        # generic fields, so it names BOTH pages. Binding to one grounds and binding
        # to the other does not, and nothing says which was meant -- so refuse.
        c = claim("Drying is the most widely used preservation method.",
                  "Globally, drying is the most widely used method for preserving foods",
                  {"kind": "zim", "id": "survivorlibrary"})
        r = G.ground_answer("Is drying widely used?", [c], PAGES)
        self.assertTrue(r["abstain"])
        self.assertEqual(len(r["kept"]), 0)
        d = r["dropped"][0]
        self.assertEqual(d["verdict"], "ambiguous")
        self.assertEqual(d["citation_matched"], 2)
        self.assertIn("changes the verdict", d["why"])

    def test_ambiguity_that_cannot_change_the_verdict_is_kept(self):
        # THE CASE THAT USED TO BE REJECTED. Two DIFFERENT pages carry the identical
        # passage -- real on this corpus: the same text appears at p.57 of two
        # printings of one chemistry manual. A citation that cannot tell them apart is
        # still correct, because the span is verbatim on every page it names and the
        # verdict is the same whichever was meant. Refusing here rejects a true claim.
        twin = dict(PAGE_SMALL)
        twin["requested"] = twin["retrieved"] = {
            "kind": "zim", "id": "survivorlibrary",
            "path": "smallscale-food-drying-technologies-2nd-printing.pdf", "page": 1}
        c = claim("Drying is the most widely used preservation method.",
                  "Globally, drying is the most widely used method for preserving foods",
                  {"kind": "zim", "id": "survivorlibrary"})
        r = G.ground_answer("Is drying widely used?", [c], [PAGE_SMALL, twin])
        self.assertTrue(r["grounded"])
        self.assertFalse(r["abstain"])
        kept = r["kept"][0]
        self.assertEqual(kept["citation_matched"], 2)
        self.assertEqual(len(kept["found_on"]), 2)   # both named, both carry the span
        self.assertFalse(kept["support_checked"])

    def test_single_match_still_records_one(self):
        # Guard against over-correcting: the ordinary unambiguous case is unchanged
        # and must not start reporting a list of pages.
        c = claim("Drying prevents microbial growth by removing moisture.",
                  "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
                  PAGE_PRESERVE["retrieved"])
        r = G.ground_answer("q", [c], PAGES)
        self.assertTrue(r["grounded"])
        self.assertEqual(r["kept"][0]["citation_matched"], 1)
        self.assertIsNone(r["kept"][0]["found_on"])

    def test_citation_matching_no_retrieved_page_is_not_grounded(self):
        # Cites a page that was never retrieved: zero matches, so no cited page.
        c = claim("Drying is the most widely used preservation method.",
                  "Globally, drying is the most widely used method for preserving foods",
                  {"kind": "zim", "id": "survivorlibrary",
                   "path": "never-retrieved.pdf", "page": 7})
        r = G.ground_answer("Is drying widely used?", [c], PAGES)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["dropped"][0]["citation_matched"], 0)

    def test_fully_specified_citation_still_grounds(self):
        # Guard against over-correcting: a precise citation must still pass.
        c = claim("Drying is the most widely used preservation method.",
                  "Globally, drying is the most widely used method for preserving foods",
                  PAGE_SMALL["retrieved"])
        r = G.ground_answer("Is drying widely used?", [c], PAGES)
        self.assertTrue(r["grounded"])
        self.assertEqual(r["kept"][0]["citation_matched"], 1)


class Undecidable(unittest.TestCase):
    """C1 has three verdicts and `abstain` is not `fail`.

    An earlier version of this module compared `c1.check(...)[0] == "pass"` and so
    reported `unfounded` — "the model made this up" — for spans C1 had explicitly
    declined to judge. Measured before the fix: a Greek-bearing span that was
    genuinely on the page came back `unfounded`. Two things were wrong with that.
    It accuses the model of something the checker never claimed, and it silently
    inflates the paraphrase rate this project reports, because a dropped claim is
    counted as the model's failure rather than the checker's limit.
    """

    def test_abstain_is_undecidable_not_unfounded(self):
        # Derived attribution: no citation to be wrong about, span really present.
        c = claim("Alpha denotes the first quantity.",
                  "Greek letter alpha, written άλφα, denotes the first", None)
        r = G.ground_answer("What is alpha?", [c], [PAGE_GREEK])
        d = r["dropped"][0]
        self.assertEqual(d["verdict"], "undecidable")
        self.assertNotEqual(d["verdict"], "unfounded")
        self.assertIn("could not decide", d["why"])

    def test_abstain_on_the_cited_page_is_undecidable(self):
        c = claim("Alpha denotes the first quantity.",
                  "Greek letter alpha, written άλφα, denotes the first",
                  PAGE_GREEK["retrieved"])
        r = G.ground_answer("What is alpha?", [c], [PAGE_GREEK] + PAGES)
        self.assertEqual(r["dropped"][0]["verdict"], "undecidable")

    def test_undecidable_still_drops_the_claim(self):
        # Not deciding is not grounding. It is dropped — just not accused.
        c = claim("Alpha denotes the first quantity.",
                  "Greek letter alpha, written άλφα, denotes the first", None)
        r = G.ground_answer("What is alpha?", [c], [PAGE_GREEK])
        self.assertTrue(r["abstain"])
        self.assertEqual(r["kept"], [])

    def test_genuinely_absent_span_is_still_unfounded(self):
        # The regression guard in the other direction: `undecidable` must not
        # become a soft landing that swallows real confabulation.
        c = claim("Freezing is best.",
                  "Freezing is the single best method for preserving every kind of food",
                  None)
        r = G.ground_answer("What is best?", [c], [PAGE_GREEK] + PAGES)
        self.assertEqual(r["dropped"][0]["verdict"], "unfounded")

    def test_one_undecidable_does_not_sink_a_grounded_answer(self):
        good = claim("Drying prevents microbial growth by removing moisture.",
                     "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
                     PAGE_PRESERVE["retrieved"])
        odd = claim("Alpha denotes the first quantity.",
                    "Greek letter alpha, written άλφα, denotes the first", None)
        r = G.ground_answer("How does drying preserve food?", [good, odd],
                            PAGES + [PAGE_GREEK])
        self.assertTrue(r["grounded"])
        self.assertEqual(len(r["kept"]), 1)
        self.assertEqual([d["verdict"] for d in r["dropped"]], ["undecidable"])


class ResolutionDrift(unittest.TestCase):
    """C1.8 — a real span in the document that came back instead of the one asked for.

    Hermetic twin of the golden-set case (`test_conformance.py::c09`), so the rule
    still has a test when cordexa's examples are not on disk.

    This layer has to check drift itself. `check()` catches it by comparing the
    claimed locator against `retrieved`, defaulting the claim to `requested` — and
    this layer overrides that default with `retrieved` on every call, because it
    computes attribution instead of trusting it. That makes the locator arm a
    tautology, so C1.8 has to be re-established here or it is simply not checked.
    """

    def drifted_page(self):
        return {"requested": {"kind": "zim", "id": "survivorlibrary", "path": "mint.pdf", "page": 1},
                "retrieved": {"kind": "zim", "id": "survivorlibrary", "path": "spearmint.pdf", "page": 1},
                "resolution": "search",
                "text": ("Spearmint oil is dominated by carvone rather than menthol, which "
                         "distinguishes it from peppermint.")}

    def test_span_in_a_drifted_page_is_not_grounded(self):
        p = self.drifted_page()
        c = claim("Mint oil is dominated by carvone rather than menthol.",
                  "dominated by carvone rather than menthol", None)
        r = G.ground_answer("What dominates mint oil?", [c], [p])
        self.assertTrue(r["abstain"])
        self.assertEqual(r["dropped"][0]["verdict"], "drifted")

    def test_drift_names_both_locators(self):
        p = self.drifted_page()
        c = claim("Mint oil is dominated by carvone rather than menthol.",
                  "dominated by carvone rather than menthol", None)
        why = G.ground_answer("q", [c], [p])["dropped"][0]["why"]
        self.assertIn("mint.pdf", why)
        self.assertIn("spearmint.pdf", why)
        self.assertIn("search", why)          # the resolution that let it through

    def test_drift_is_not_confabulation(self):
        # The words are real. Blaming the model for the retriever's miss is the
        # same error class as the abstain collapse.
        p = self.drifted_page()
        c = claim("Mint oil is dominated by carvone rather than menthol.",
                  "dominated by carvone rather than menthol", None)
        self.assertNotEqual(G.ground_answer("q", [c], [p])["dropped"][0]["verdict"], "unfounded")

    def test_absent_span_on_a_drifted_page_is_still_unfounded(self):
        # Only a PASS is downgraded — otherwise drift would hide plain confabulation.
        p = self.drifted_page()
        c = claim("Mint cures scurvy.",
                  "Mint has long been the accepted treatment for scurvy at sea", None)
        self.assertEqual(G.ground_answer("q", [c], [p])["dropped"][0]["verdict"], "unfounded")

    def test_undrifted_pages_are_unaffected(self):
        # Guard against over-correcting: requested == retrieved must still ground.
        c = claim("Drying prevents microbial growth by removing moisture.",
                  "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
                  None)
        self.assertTrue(G.ground_answer("q", [c], PAGES)["grounded"])

    def test_extra_fields_on_retrieved_are_not_drift(self):
        # A real adapter's `retrieved` carries url/score the request never had.
        p = dict(PAGE_PRESERVE)
        p["retrieved"] = dict(PAGE_PRESERVE["retrieved"], url="file:///x.pdf", score=0.81)
        c = claim("Drying prevents microbial growth by removing moisture.",
                  "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
                  None)
        self.assertTrue(G.ground_answer("q", [c], [p])["grounded"])


class CrossPage(unittest.TestCase):
    """A quotation that runs across a page break.

    Measured on the corpus: 48.7% of adjacent page pairs cut a sentence. Against
    single-page documents C1 finds such a span in neither page and the claim comes
    back `unfounded` -- a real passage reported as confabulated, the same shape as
    the abstain collapse. corpus.hits_to_pages hands over pages JOINED where the
    text continues; this is what the verifier must then do with them.
    """

    def pages(self):
        def pg(n, text):
            i = {"kind": "zim", "doc": "mill.pdf", "page": n}
            return {"requested": i, "retrieved": i, "resolution": "exact", "text": text}
        a = pg(300, "The circular saw must be kept true, and the operator should never")
        b = pg(301, "attempt to clear a jam while the arbor is still turning.")
        i = {"kind": "zim", "doc": "mill.pdf", "page": 300, "page_to": 301}
        return a, b, {"requested": i, "retrieved": i, "resolution": "exact",
                      "text": a["text"] + "\n" + b["text"], "parts": [a, b]}

    def test_unjoined_pages_call_a_real_passage_unfounded(self):
        # The bug this exists to prevent, asserted so it cannot come back.
        a, b, _ = self.pages()
        span = "the operator should never attempt to clear a jam while the arbor"
        r = G.ground_answer("q", [claim("c", span, None)], [a, b])
        self.assertEqual(r["dropped"][0]["verdict"], "unfounded")

    def test_joined_pages_ground_it(self):
        _, _, joined = self.pages()
        span = "the operator should never attempt to clear a jam while the arbor"
        r = G.ground_answer("q", [claim("c", span, None)], [joined])
        self.assertTrue(r["grounded"])
        self.assertEqual(r["kept"][0]["cited_page"]["page"], 300)
        self.assertEqual(r["kept"][0]["cited_page"]["page_to"], 301)

    def test_a_span_on_one_page_narrows_to_that_page(self):
        # Citing p.300-301 for something wholly on p.300 is a worse citation than
        # the index can support, so the range is narrowed away.
        _, _, joined = self.pages()
        span = "The circular saw must be kept true, and the operator"
        r = G.ground_answer("q", [claim("c", span, None)], [joined])
        cp = r["kept"][0]["cited_page"]
        self.assertEqual(cp["page"], 300)
        self.assertNotIn("page_to", cp)

    def test_narrowing_picks_the_second_page_too(self):
        _, _, joined = self.pages()
        span = "attempt to clear a jam while the arbor is still turning"
        cp = G.ground_answer("q", [claim("c", span, None)], [joined])["kept"][0]["cited_page"]
        self.assertEqual(cp["page"], 301)
        self.assertNotIn("page_to", cp)

    def test_a_confabulated_span_is_still_unfounded_on_a_join(self):
        # Joining must not become a way for made-up text to find a home.
        _, _, joined = self.pages()
        r = G.ground_answer("q", [claim("c", "the operator should wear a bright red hat", None)],
                            [joined])
        self.assertEqual(r["dropped"][0]["verdict"], "unfounded")


if __name__ == "__main__":
    unittest.main()
