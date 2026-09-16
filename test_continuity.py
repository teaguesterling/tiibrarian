"""Page-continuation detection. Hermetic.

    python3 -m unittest test_continuity -v
"""
import unittest

import continuity as C

# Fixtures with CONTROLLED widths, so the fill arithmetic is checkable rather
# than accidental. The first version of this file used prose whose last line had
# no full stop and a "ragged" tail of three words -- which prose_lines() drops as
# furniture -- so it tested neither thing it claimed to.
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "


def line(n, end=""):
    """A prose line of ~n characters, Capitalised, optionally ending with `end`.

    Capitalised because WORDS starts with "alpha": a lowercase first character
    made every fixture page trip the "next page opens lowercase" rule, so three
    tests failed for a reason that had nothing to do with what they tested.
    """
    body = (WORDS * 8)[:max(n - len(end), 12)].rstrip()
    return body[:1].upper() + body[1:] + end


W = 72                                  # column width for these fixtures
BODY = "\n".join(line(W) for _ in range(3))


def page(*extra, body=BODY):
    return body + ("\n" + "\n".join(extra) if extra else "")


FULL_TAIL = line(68, ".")               # 68/72 = 0.94 -> reaches the margin
RAGGED_TAIL = line(30, ".")             # 30/72 = 0.42 -> the paragraph ended
CUT_TAIL = line(68)                     # no terminal punctuation at all


class Furniture(unittest.TestCase):
    """Page numbers and running heads are not prose. Comparing them across a
    break is what made the first measurement wrong by 27x."""

    def test_page_number_is_not_prose(self):
        self.assertEqual(C.prose_lines("312"), [])

    def test_running_head_is_not_prose(self):
        self.assertEqual(C.prose_lines("THE SAWMILL AND ITS CARE"), [])

    def test_short_line_is_not_prose(self):
        self.assertEqual(C.prose_lines("Fig. 12"), [])

    def test_body_text_is_prose(self):
        self.assertEqual(len(C.prose_lines(BODY)), 3)

    def test_the_fixtures_are_what_they_claim(self):
        # guards the test file against itself
        self.assertEqual(C.column_width(C.prose_lines(page(FULL_TAIL))), W)
        self.assertGreaterEqual(len(FULL_TAIL) / W, C.FILL_FULL)
        self.assertLess(len(RAGGED_TAIL) / W, C.FILL_FULL)
        self.assertTrue(C.ends_sentence(FULL_TAIL))
        self.assertTrue(C.ends_sentence(RAGGED_TAIL))
        self.assertFalse(C.ends_sentence(CUT_TAIL))
        for t in (FULL_TAIL, RAGGED_TAIL, CUT_TAIL):
            self.assertEqual(len(C.prose_lines(t)), 1, "%r must count as prose" % t)

    def test_furniture_does_not_become_the_tail(self):
        # The real last prose line must win over a trailing page number.
        got = C.prose_lines(page(FULL_TAIL, "312"))
        self.assertEqual(got[-1], FULL_TAIL)


class APeriodIsNotAFullStop(unittest.TestCase):
    """Abbreviations end a line with a dot without ending a sentence, and this
    corpus is made of them. Reading "6 ft." as a full stop sends the detector to
    the geometry path when it should already know the text continues."""

    def test_abbreviations_do_not_end_a_sentence(self):
        for t in ("and so forth, etc.", "a depth of 6 ft.", "the works at St.",
                  "see vol.", "roughly 40 lb.", "printed by Bros."):
            self.assertFalse(C.ends_sentence(t), t)

    def test_initials_do_not_end_a_sentence(self):
        self.assertFalse(C.ends_sentence("a letter from J."))
        self.assertFalse(C.ends_sentence("the U.S."))

    def test_real_sentence_ends_still_do(self):
        for t in ("It stopped here.", "the work was done!", "was it finished?",
                  "by J. S. Woodward.", "the U.S. Navy.", 'he said "stop."'):
            self.assertTrue(C.ends_sentence(t), t)

    def test_an_abbreviation_at_the_break_means_continue(self):
        # The whole point: this used to fall through to geometry and could be
        # judged "ragged", refusing a join across an obvious continuation.
        # `line(n, end)` appends `end` with no space, which is right for a
        # full stop and wrong for a word -- so build this one explicitly.
        a = page(line(26) + " etc.")
        ok, why = C.continues(a, page())
        self.assertTrue(ok, why)
        self.assertIn("mid-sentence", why)


class Capitalisation(unittest.TestCase):
    """An uppercase opening is evidence of NOTHING. Proper nouns start pages."""

    def test_proper_noun_head_does_not_block_a_join(self):
        # tail is cut mid-sentence; the next page opens "Sheffield". The tail
        # rule must decide this, not the capital letter.
        ok, why = C.continues(page(CUT_TAIL), "Sheffield was reached by noon and "
                                              "the party went on to the mill\n" + BODY)
        self.assertTrue(ok)
        self.assertIn("mid-sentence", why)

    def test_uppercase_head_is_not_evidence_either_way(self):
        # It must not be read as "new sentence" -- but with the sentence already
        # finished and no boxes, there is nothing left to decide on, so decline.
        ok, why = C.continues(page(FULL_TAIL), page())
        self.assertFalse(ok)
        self.assertNotIn("lowercase", why)


class SentenceCut(unittest.TestCase):
    def test_unterminated_tail_continues(self):
        ok, why = C.continues(page(CUT_TAIL), page())
        self.assertTrue(ok)
        self.assertIn("mid-sentence", why)

    def test_lowercase_head_continues(self):
        a = page(FULL_TAIL)
        b = "attempt to clear a jam while the arbor is still turning at speed\n" + BODY
        ok, why = C.continues(a, b)
        self.assertTrue(ok)
        self.assertIn("lowercase", why)


class ParagraphGeometry(unittest.TestCase):
    """A completed sentence at a break is ambiguous in plain text; only the
    line's fill separates a new paragraph from a continuing one."""

    def test_a_finished_sentence_does_not_join_without_geometry(self):
        """Plain text cannot tell a new paragraph from a continuing one.

        A full-width last line used to be read as "the paragraph runs on". An
        audit of 22 sampled decisions found that rule right 0 times out of 2: it
        joined a finished section to the next one, and joined prose to a verse
        epigraph. It was wrong the other way too -- verse is ragged by
        construction, so a couplet continuing across a break read as finished.
        Guessing lost both ways, so there is no guess: decline.
        """
        for tail in (FULL_TAIL, RAGGED_TAIL):
            ok, why = C.continues(page(tail), page())
            self.assertFalse(ok, "%r -> %s" % (tail, why))
            self.assertIn("no geometry", why)

    def test_the_case_worth_protecting(self):
        # 19.8% of breaks. Joining here would manufacture a passage that spans a
        # chapter end -- text that is not contiguous in the book.
        end_of_chapter = page(RAGGED_TAIL)
        start_of_next = "CHAPTER IV\n" + BODY
        self.assertFalse(C.continues(end_of_chapter, start_of_next)[0])


class Conservative(unittest.TestCase):
    def test_no_prose_does_not_join(self):
        ok, why = C.continues("312", page())
        self.assertFalse(ok)
        self.assertIn("no prose", why)

    def test_empty_does_not_join(self):
        self.assertFalse(C.continues("", "")[0])
        self.assertFalse(C.continues(None, None)[0])

    def test_a_table_page_does_not_join(self):
        # Numeric tables have no prose lines at all; joining one to the prose
        # after it would invent adjacency across a real discontinuity.
        table = "\n".join(["1000 000 000 043 087 434 867 477 911 344 777 209",
                           "1001 000 434 867 477 911 344 777 209 130 173 217"])
        self.assertFalse(C.continues(table, page())[0])


class Boxes(unittest.TestCase):
    """When a page carries OCR boxes, geometry replaces character counting."""

    def _boxes(self, rights, left=100):
        # one quad per line: [x0,y0, x1,y0, x1,y1, x0,y1]
        return [[left, i * 20, r, i * 20, r, i * 20 + 15, left, i * 20 + 15]
                for i, r in enumerate(rights)]

    def test_line_reaching_the_margin_continues(self):
        ok, why = C.continues(page(FULL_TAIL), page(),
                              boxes_a=self._boxes([900, 900, 900, 895]), page_a=1)
        self.assertTrue(ok, why)
        self.assertIn("boxes", why)
        # and without the boxes the same pair declines -- geometry is the ONLY
        # thing that can decide a finished sentence.
        self.assertFalse(C.continues(page(FULL_TAIL), page())[0])

    def test_line_stopping_short_does_not(self):
        ok, why = C.continues(page(RAGGED_TAIL), page(),
                              boxes_a=self._boxes([900, 900, 900, 300]), page_a=1)
        self.assertFalse(ok)
        self.assertIn("boxes", why)

    def test_indented_next_page_is_a_new_paragraph(self):
        a = page(FULL_TAIL)
        indented = self._boxes([900, 900, 900])
        indented[0][0] = indented[0][6] = 160        # first line pushed right
        ok, why = C.continues(a, page(), boxes_a=self._boxes([900, 900, 900, 898]),
                              boxes_b=indented, page_a=1)
        self.assertFalse(ok)
        self.assertIn("indented", why)

    def test_boxes_win_over_the_text_heuristic(self):
        """Geometry is the better evidence and must override character counting.

        A short line that nonetheless reached the right margin -- a narrow
        column, a wide-set final line -- reads as ragged by character count and
        as continuing by geometry. Geometry is right.
        """
        a = page(RAGGED_TAIL)
        self.assertFalse(C.continues(a, page())[0])            # text says stop
        ok, why = C.continues(a, page(),
                              boxes_a=self._boxes([900, 900, 900, 890]), page_a=1)
        self.assertTrue(ok, why)                               # boxes say go on
        self.assertIn("boxes", why)


if __name__ == "__main__":
    unittest.main()
