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
    "any.docx#duck-blocks:12-40",                 # block range (format-agnostic)
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
        self.assertEqual(L.kind(L.parse("a.docx#duck-blocks:1-9")), "blocks")
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


class BlockRangeSpelling(unittest.TestCase):
    """The block range is namespaced because it is the one form we invented."""

    def test_three_spellings_are_one_locator(self):
        # Raw duck, percent-encoded duck, and the readable name must be EQUAL, not
        # merely all parseable: two records citing one passage have to compare equal
        # or a dedup pass silently keeps both.
        a = L.parse("a.docx#duck-blocks:12-40")
        b = L.parse("a.docx#\U0001F986:12-40")
        c = L.parse("a.docx#%F0%9F%A6%86:12-40")
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_the_duck_is_an_ALIAS_not_a_second_canonical_form(self):
        # Asserted explicitly: "fun shortcut" is exactly the kind of thing that
        # silently becomes canonical. render() emits the readable spelling so stored
        # locators stay greppable.
        loc = L.parse("a.docx#\U0001F986:1-9")
        self.assertEqual(L.render(loc), "a.docx#duck-blocks:1-9")
        self.assertEqual(L.render(loc, duck=True), "a.docx#\U0001F986:1-9")

    def test_a_real_id_shaped_like_the_old_form_is_a_section(self):
        # THE WHOLE POINT. A document may genuinely contain <div id="b12-b40">;
        # numbered boxes and lettered clauses produce ids like that. It must resolve
        # as a section, which the old bare spelling made impossible.
        self.assertEqual(L.kind(L.parse("a.html#b12-b40")), "section")
        self.assertEqual(L.parse("a.html#b12-b40")["section"], "b12-b40")

    def test_the_namespace_colon_does_not_collide_with_a_scheme(self):
        # zim:// and duck-blocks: both contain ':' and mean different things.
        loc = L.parse("zim://w.zim/Entry#duck-blocks:3-9")
        self.assertEqual(loc["doc"], "zim://w.zim/Entry")
        self.assertEqual((loc["blocks"], loc["blocks_to"]), (3, 9))


class EmittedCallsCarryTheDocument(unittest.TestCase):
    """The reader gets the document, fragment stripped -- never the whole locator."""

    def test_the_zim_entry_keeps_its_hash_but_loses_the_fragment(self):
        # DISCRIMINATING, and only a C# article would ever catch it: duckeye takes
        # everything after "<archive>.zim/" as the entry name VERBATIM and does no
        # fragment parsing, so handing it the whole locator asks for an entry literally
        # named "C#Sharp#overview", which does not exist. The entry's own '#' must
        # survive; ours must not.
        call = L.how_to_read(L.parse("zim://w.zim/C#Sharp#overview"))[0]
        self.assertIn("'zim://w.zim/C#Sharp'", call)
        self.assertNotIn("overview'", call.split(",")[0])
        self.assertIn("'overview'", call)

    def test_no_placeholder_survives(self):
        # A literal `src` in the output invites a caller to substitute the whole
        # locator string, which is the mistake this exists to prevent.
        for spec in ("a.pdf#p.4", "a.html#intro", "a.docx#duck-blocks:1-9",
                     "a.md#L2-L5", "a.txt"):
            for call in L.how_to_read(L.parse(spec)):
                self.assertNotIn("(src", call, spec)

    def test_a_quote_in_the_path_is_escaped(self):
        call = L.how_to_read(L.parse("o'brien.pdf#p.3"))[0]
        self.assertIn("'o''brien.pdf'", call)


class Resolvability(unittest.TestCase):
    def test_resolvable_forms_name_a_real_call(self):
        for s, frag in (("a.pdf#p.4", "read_pdf_blocks"),
                        ("a.html#intro", "doc_section"),
                        ("a.docx#duck-blocks:1-9", "duck_blocks_slice"),
                        ("a.txt", "panduck_read_blocks")):
            self.assertTrue(L.resolvable(L.parse(s)), s)
            calls = " ".join(L.how_to_read(L.parse(s)))
            self.assertIn(frag, calls, s)
            self.assertIn("'" + s.split("#")[0] + "'", calls, s)   # the document, quoted

    def test_a_named_fragment_resolves_on_two_axes_in_order(self):
        # doc_section walks HEADINGS; doc_container walks the STRUCTURAL nesting. A
        # fragment on <h2 id="methods"> wants the first, one on <div id="sidebar"> the
        # second -- and doc_section returns NOTHING for a div, so a single-resolver
        # grammar silently fails to resolve half the fragments a browser handles.
        calls = L.how_to_read(L.parse("page.html#sidebar"))
        self.assertEqual(len(calls), 2)
        self.assertIn("doc_section", calls[0])
        self.assertIn("doc_container", calls[1])

    def test_positional_forms_have_exactly_one_resolution(self):
        # Only named fragments are ambiguous about axis; a page is a page.
        for s in ("a.pdf#p.4", "a.docx#duck-blocks:1-9", "a.txt"):
            self.assertEqual(len(L.how_to_read(L.parse(s))), 1, s)

    def test_line_ranges_resolve_through_read_lines(self):
        # read_lines(src) yields line_number/content/byte_offset -- the extension
        # duckeye already installs for its -f lines mode -- so a line range is a
        # BETWEEN on line_number. It returns text lines rather than duck_blocks,
        # which is fine for grounding: C1 checks a span against text.
        loc = L.parse("notes.md#L2-L5")
        self.assertEqual((loc["lines"], loc["lines_to"]), (2, 5))
        self.assertTrue(L.resolvable(loc))
        call = L.how_to_read(loc)[0]
        self.assertIn("read_lines", call)
        self.assertIn("BETWEEN 2 AND 5", call)

    def test_a_single_line_is_a_degenerate_range(self):
        self.assertIn("BETWEEN 7 AND 7", L.how_to_read(L.parse("a.md#L7"))[0])

    def test_id_capture_limitation_is_recorded(self):
        # Whether a #name can resolve depends on the INSTALLED webbed, not on panduck:
        # the published build keeps id only on div, section/article and headings, so
        # <ul id="steps"> is unaddressable today. Recorded rather than discovered at
        # query time.
        self.assertIn("webbed", L.ID_CAPTURE)


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
