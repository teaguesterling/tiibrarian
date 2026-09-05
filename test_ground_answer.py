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


if __name__ == "__main__":
    unittest.main()
