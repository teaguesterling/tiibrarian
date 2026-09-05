"""Locator parsing, rendering and resolvability. Hermetic."""
import unittest

import ground_answer as G
import locator as L

ROUNDTRIP = [
    "plain.txt",                                  # whole document
    "manual.pdf#p.27",                            # one page
    "manual.pdf#p.3-7",                           # page range
    "foo.html#methods",                           # section by id
    "book.epub#chapter-3",
    "any.docx#b12-b40",                           # block range (format-agnostic)
    "notes.md#L2-L5",                             # line range (not resolvable yet)
    "zim://wiki.zim/Photosynthesis#intro",        # duckeye zim scheme
    "git://README.md@HEAD~1#Install",             # duckeye git scheme, rev has no '#'
]


class Roundtrip(unittest.TestCase):
    def test_every_form_round_trips(self):
        for s in ROUNDTRIP:
            self.assertEqual(L.render(L.parse(s)), s, s)

    def test_kinds(self):
        self.assertEqual(L.kind(L.parse("a.pdf#p.4")), "page")
        self.assertEqual(L.kind(L.parse("a.md#L2-L5")), "lines")
        self.assertEqual(L.kind(L.parse("a.docx#b1-b9")), "blocks")
        self.assertEqual(L.kind(L.parse("a.html#intro")), "section")
        self.assertIsNone(L.kind(L.parse("a.txt")))

    def test_fragment_splits_on_the_LAST_hash(self):
        # A zim entry or URL may contain '#'; the fragment is the one we appended.
        loc = L.parse("zim://w.zim/C#Sharp#overview")
        self.assertEqual(loc["doc"], "zim://w.zim/C#Sharp")
        self.assertEqual(loc["section"], "overview")

    def test_a_bare_document_is_a_legal_locator(self):
        self.assertEqual(L.parse("a.txt"), {"doc": "a.txt"})

    def test_empty_is_rejected(self):
        for bad in ("", None, 7):
            with self.assertRaises(ValueError):
                L.parse(bad)


class Resolvability(unittest.TestCase):
    def test_resolvable_forms_name_a_real_call(self):
        for s, frag in (("a.pdf#p.4", "read_pdf_blocks"),
                        ("a.html#intro", "doc_section"),
                        ("a.docx#b1-b9", "duck_blocks_slice"),
                        ("a.txt", "panduck_read_blocks")):
            self.assertTrue(L.resolvable(L.parse(s)), s)
            self.assertIn(frag, L.how_to_read(L.parse(s)))

    def test_line_ranges_parse_but_are_honestly_unresolvable(self):
        # #L2-L5 is the natural spelling for Markdown and nothing in the stack can
        # read it back yet. It must not claim otherwise.
        loc = L.parse("notes.md#L2-L5")
        self.assertEqual((loc["lines"], loc["lines_to"]), (2, 5))
        self.assertFalse(L.resolvable(loc))
        self.assertIsNone(L.how_to_read(loc))


PAGE = {"requested": {"kind": "corpus", "doc": "manual.pdf", "page": 27},
        "retrieved": {"kind": "corpus", "doc": "manual.pdf", "page": 27},
        "resolution": "exact",
        "text": "Blanch the beans for three minutes before packing them into jars."}
OTHER = {"requested": {"kind": "corpus", "doc": "other.pdf", "page": 5},
         "retrieved": {"kind": "corpus", "doc": "other.pdf", "page": 5},
         "resolution": "exact", "text": "Unrelated prose about radio valves."}


class StringCitations(unittest.TestCase):
    """A model writes `manual.pdf#p.27`, not a dict."""

    def test_a_locator_string_grounds(self):
        r = G.ground_answer("q", [{"text": "c", "span": "Blanch the beans for three minutes",
                                   "cites": "manual.pdf#p.27"}], [PAGE, OTHER])
        self.assertTrue(r["grounded"])
        self.assertEqual(r["kept"][0]["citation_matched"], 1)

    def test_a_locator_string_naming_the_wrong_page_is_misattributed(self):
        r = G.ground_answer("q", [{"text": "c", "span": "Blanch the beans for three minutes",
                                   "cites": "other.pdf#p.5"}], [PAGE, OTHER])
        self.assertTrue(r["abstain"])
        self.assertEqual(r["dropped"][0]["verdict"], "misattributed")

    def test_a_document_only_string_still_resolves_when_it_names_one_page(self):
        r = G.ground_answer("q", [{"text": "c", "span": "Blanch the beans for three minutes",
                                   "cites": "manual.pdf"}], [PAGE, OTHER])
        self.assertTrue(r["grounded"])   # 'doc' alone picks out exactly one retrieved page

    def test_an_unparseable_citation_does_not_crash_or_silently_match(self):
        r = G.ground_answer("q", [{"text": "c", "span": "Blanch the beans for three minutes",
                                   "cites": "###"}], [PAGE, OTHER])
        self.assertTrue(r["abstain"])


if __name__ == "__main__":
    unittest.main()
