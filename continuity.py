"""Does the text on one page run on into the next?

A quoted passage does not respect page breaks. Measured with THIS detector over
40,000 adjacent page pairs with prose on both sides:

    80.2%   the text continues across the break
      74.0%   tail ends mid-sentence
       6.2%   next page opens lowercase
    19.8%   the text stops, or cannot be judged from text alone

So continuing is the overwhelmingly common case, and a verifier that only ever
sees one page's text calls every cross-page quotation `unfounded` -- measured,
and the same false-accusation shape as reading C1's `abstain` as confabulation.

AUDITED BY READING, not just counted. 22 sampled decisions, both directions:

    rule                        share   correct in sample
    tail ends mid-sentence      74.0%   6/6
    ragged -> decline           19.8%   ~8/11
    fill -> join (REMOVED)       6.3%   0/2

The mid-sentence rule is reliable, including the case that most needs it: a word
hyphenated across the break ("...Shreveport. An-" / "other trunk line..."). The
fill rule -- join when the last line runs the full column -- was wrong every time
it was checked. It joined a finished section to the next one, and joined prose to
a verse epigraph. So it is gone: when the sentence ended and the next page opens
with a capital, plain text cannot tell a new paragraph from a continuing one, and
declining is the side that only costs recall.

KNOWN RECALL LOSSES, both in the safe direction and both found in that audit:

  * VERSE. Poetry lines are ragged by construction, so a couplet continuing
    across a break reads as a finished paragraph and is declined.
  * FOOTNOTES. `prose_lines` does not recognise them, so a footnote at the foot
    of a page becomes the "tail" and hides body text that really was cut. Seen:
    "1 Elements of Railroad Economics, p. 75." standing in for a sentence ending
    "...developing commerce and industry with".

Both lose joins rather than inventing them, which is the tolerable direction.

TWO EARLIER MEASUREMENTS WERE WRONG, both because they were SQL
re-implementations rather than this code:

  * the first compared the last LINE of a page to the first LINE of the next --
    usually a footer and a running head -- and reported 1.8%.
  * the second stripped furniture but scored a break as continuing only when the
    next page opened LOWERCASE, and reported 48.7%. Capitalisation is not
    evidence of a new sentence: proper nouns, initials and abbreviations open
    pages constantly. That rule scores 49.2% on the same sample this detector
    scores 80.2% on.

Measure with the code that ships. Both errors were invisible in their own terms
and only showed up against a differently-built number.

The fix is to hand the verifier the pages joined. The question this module
answers is WHICH pairs to join, and the reason to ask is not economy -- at 80.2%
conditional joining saves little work. It is the other 19.8%: at a chapter end, a
section break, or a table followed by prose, an unconditional join MANUFACTURES A
PASSAGE THAT IS NOT IN THE BOOK, and C1 would ground a span quoted across that
seam. Never inventing adjacency is worth more than the recall.

A PERIOD IS NOT A FULL STOP. This corpus is technical prose full of "No. 5.",
"Fig. 12.", "6 ft.", "etc." and "J. S. Woodward" -- see `ends_sentence`. Reading
those as sentence ends sends the detector to the geometry path when it should
already have concluded the text continues.

PARAGRAPHS, NOT SENTENCES. A quoted span can cross a sentence boundary inside a
paragraph, so joining only on a cut sentence would still under-join. A completed
sentence at a page break is ambiguous in plain text -- a new paragraph and a
continuing one look identical -- and fill is the only text-only signal that
separates them.

WITH BOXES IT GETS BETTER. Every page OCR'd from 2026-09-05 carries per-line
boxes, so "did this line reach the right margin" and "is the next page's first
line indented" become geometry instead of character counting. `continues()` uses
them when a page has them and falls back to the text heuristics when it does not,
which is most of the corpus until the backfill runs.
"""
import re

TERMINAL = re.compile(r'[.!?]["\'”’)\]]?\s*$')
_LAST_TOKEN = re.compile(r'([A-Za-z][A-Za-z.]*)\.["\'”’)\]]?\s*$')

# A period is not a full stop. These end in one constantly, mid-sentence.
ABBREV = {
    "etc", "vs", "viz", "cf", "ibid", "op", "cit", "al", "eg", "ie",
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev", "hon",
    "gen", "col", "capt", "lt", "sgt", "maj", "adm", "supt",
    "no", "nos", "fig", "figs", "vol", "vols", "ch", "chap", "sec", "art",
    "pp", "ed", "eds", "trans", "rev", "approx", "dept", "est", "inc",
    "ltd", "co", "corp", "bros", "ave", "blvd", "mt", "rd",
    "ft", "in", "yd", "lb", "lbs", "oz", "hr", "hrs", "min", "sq", "cu", "deg",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
}


def ends_sentence(line):
    """Does this line actually END a sentence, or just happen to stop on a dot?

    `[.!?]$` is not the question. This corpus is technical prose from the 1800s
    and 1900s: "No. 5.", "Fig. 12.", "J. S. Woodward", "6 ft.", "etc." all end a
    LINE with a period without ending a sentence, and reading them as full stops
    sends the detector down the paragraph-geometry path when it should already
    have said "this continues".
    """
    s = (line or "").rstrip()
    if not TERMINAL.search(s):
        return False
    if s.rstrip("\"'”’)]").endswith(("!", "?")):
        return True
    m = _LAST_TOKEN.search(s)
    if not m:
        return True                      # "...5." — a number, treat as a stop
    tok = m.group(1)
    if len(tok) == 1 and tok.isupper():
        return False                     # an initial: "J."
    if "." in tok:
        return False                     # "U.S.", "i.e.", "A.M."
    return tok.lower() not in ABBREV


FILL_FULL = 0.90        # >= this much of the column width means the line ran on
MIN_PROSE_ALPHA = 12
MIN_PROSE_WORDS = 4
INDENT_FRAC = 0.04      # first line indented by >4% of column width = new paragraph


def prose_lines(text):
    """Lines that look like body text, not page furniture.

    Page numbers, running heads and short all-caps titles are not prose, and
    comparing them across a break measures the furniture rather than the text.
    A first pass at this did exactly that and reported 1.8% of pages continuing,
    against a real 80.2% -- because it was reading footers.
    """
    out = []
    for line in (text or "").split("\n"):
        s = line.strip()
        if len(re.sub(r"[^A-Za-z]", "", s)) < MIN_PROSE_ALPHA:
            continue
        if len(s.split()) < MIN_PROSE_WORDS:
            continue
        if re.fullmatch(r"[0-9IVXLCivxlc.,\- ]+", s):
            continue
        if s.upper() == s and len(s) < 45:
            continue
        out.append(line)
    return out


def column_width(lines):
    """The page's column width, as the 90th percentile prose line length."""
    if not lines:
        return 0
    lens = sorted(len(x.strip()) for x in lines)
    return lens[min(int(0.9 * len(lens)), len(lens) - 1)]


def continues(text_a, text_b, boxes_a=None, boxes_b=None, page_a=None):
    """Does `text_a` run on into `text_b`? -> (bool, reason).

    Conservative on purpose: when the page has no prose to judge, the answer is
    NO. A missed join costs one claim a `drifted`-shaped miss that the caller can
    retry; a wrong join fabricates a passage and can ground a quotation that
    never existed as contiguous text.
    """
    a, b = prose_lines(text_a), prose_lines(text_b)
    if not a or not b:
        return False, "no prose on one side of the break — not joining blind"

    tail, head = a[-1].strip(), b[0].strip()

    if not ends_sentence(tail):
        return True, "tail ends mid-sentence"

    # A lowercase opening is evidence FOR continuation. An uppercase one is
    # evidence of NOTHING -- proper nouns, initials and abbreviations start
    # pages constantly, and "Sheffield was reached by noon" is just as likely to
    # be the middle of a sentence as the start of one. So the uppercase case
    # falls through to geometry rather than concluding "new sentence".
    if head[:1].islower():
        return True, "next page opens lowercase"

    # Sentence completed at the break. Only geometry can tell a new paragraph
    # from a continuing one.
    if boxes_a and page_a is not None:
        right = _right_edge(boxes_a)
        last = _line_right(boxes_a, -1)
        if right and last and last / right >= FILL_FULL:
            if boxes_b and _is_indented(boxes_b):
                return False, "next page's first line is indented — new paragraph"
            return True, "tail line reaches the right margin (boxes)"
        return False, "tail line stops short of the margin (boxes)"

    # NO TEXT-ONLY GUESS HERE. `_fill` used to join when the last line ran the
    # full column width. Audited by reading 22 sampled decisions: the
    # mid-sentence rule was right 6/6, and fill was right 0/2 -- it joined
    # "...compete favorably for remunerative traffic." to a new section, and
    # joined prose to a verse epigraph. It was wrong in the other direction too,
    # since verse is ragged by construction and a continuing couplet reads as a
    # finished paragraph.
    #
    # When the sentence ended and the next page opens with a capital, plain text
    # does not say whether the paragraph continues. Guessing loses both ways, so
    # decline: a missed join costs one claim, a wrong join fabricates a passage.
    return False, "sentence ended and no geometry to judge the paragraph"


def _right_edge(boxes):
    """The column's right edge: the furthest right any line reaches."""
    xs = [max(q[0::2]) for q in boxes if q]
    return max(xs) if xs else 0


def _line_right(boxes, i):
    q = [x for x in boxes if x]
    return max(q[i][0::2]) if q else 0


def _is_indented(boxes):
    """Is the first line pushed right of the column's left margin?"""
    q = [x for x in boxes if x]
    if len(q) < 3:
        return False
    lefts = [min(x[0::2]) for x in q]
    margin = min(lefts)
    width = _right_edge(boxes) - margin
    return width > 0 and (lefts[0] - margin) / width > INDENT_FRAC
