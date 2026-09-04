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

    def test_underspecified_citation_matching_many_pages_is_not_grounded(self):
        # The span IS verbatim on PAGE_SMALL, but the citation carries only the
        # generic fields, so it agrees with BOTH pages and identifies neither.
        # Binding it to whichever page happened to come first would hand a free
        # pass to the very misattribution this layer exists to catch.
        c = claim("Drying is the most widely used preservation method.",
                  "Globally, drying is the most widely used method for preserving foods",
                  {"kind": "zim", "id": "survivorlibrary"})
        r = G.ground_answer("Is drying widely used?", [c], PAGES)
        self.assertTrue(r["abstain"])
        self.assertEqual(len(r["kept"]), 0)
        d = r["dropped"][0]
        self.assertNotEqual(d["verdict"], "grounded")
        self.assertEqual(d["citation_matched"], 2)
        self.assertIn("under-specified", d["why"])

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
