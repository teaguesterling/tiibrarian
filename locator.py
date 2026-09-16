"""Locators: naming a passage precisely enough to cite it, in any document format.

A citation is only worth anything if the thing it names can be READ BACK. So a locator
here is not free-form: every form below maps to a call that panduck or duck_block_utils
can actually resolve, and the ones that do not resolve yet are marked as such rather
than quietly accepted.

    doc                      the document -- a path, or a duckeye URI
                             (zim://archive.zim/entry, git://path@rev, http(s)://, s3://)
    #fragment                the position inside it

FRAGMENT GRAMMAR, and what re-reads each one:

    #p.27        #p.3-7      page / page range   -> read_pdf_blocks(src, pages := '3-7')
    #duck-blocks:12-40       block range         -> duck_blocks_slice(blocks, 12, 40)
    #🦆:12-40                the same, in duck      -> ditto
    #L2-L5                   line range          -> read_lines, filtered on line_number
    #methods                 id or heading text  -> doc_section, THEN doc_container

A NAMED FRAGMENT HAS TWO RESOLUTIONS, on different axes, and which is right depends on
what the id is attached to:

  doc_section(src, id)    walks HEADINGS and bounds on heading_level -- "the prose under
                          this title". Right for <h2 id="methods">.
  doc_container(src, id)  walks the STRUCTURAL nesting in `level` -- "what is inside this
                          box": a div, section, article, list, blockquote. Right for
                          <div id="sidebar">, for which doc_section returns nothing at
                          all, since it only matches headings.

A document can disagree about the two -- a div can hold three headings, and a heading's
section can run across several divs -- so neither subsumes the other. Trying section and
falling back to container is closer to what a browser does with a fragment than either
alone, so that is the chain.

`#duck-blocks:…` is the format-agnostic one: every duck_block carries `element_order`
whatever the reader was, so a block range addresses HTML, Markdown, DOCX, EPUB and PDF
identically.

IT IS NAMESPACED ON PURPOSE. `#p.27` and `#L2-L5` are established spellings that predate
us -- page references and GitHub's line anchors -- so a document is unlikely to carry an
id shaped like either. A block range is OUR invention, and a bare `#b12-b40` is a
perfectly plausible real element id. Namespacing it means a document that genuinely
contains `<div id="b12-b40">` still resolves as a section, which is what the author meant.
`#🦆:12-40` is an ACCEPTED ALIAS, not a second canonical form: `render()` emits
`duck-blocks:` unless asked for the duck, so stored locators stay greppable and two
citations of one passage compare equal. The percent-encoded spelling
`#%F0%9F%A6%86:12-40` parses to the same thing, since a fragment is percent-encoded on
the wire.

The bare `#b12-b40` form this replaces is GONE, not deprecated. Keeping it would defeat
the change: a document that really contains `<div id="b12-b40">` must resolve as a
section, and it now does. Nothing had recorded one -- the corpus emits page locators
only -- so there is no migration to do.
Pages only exist in paginated formats; sections only where there are headings; lines only
in text-ish sources. Prefer the most specific form the format actually supports, and fall
back to `#b…`, which always exists.

LINES RESOLVE THROUGH A DIFFERENT SHAPE, and that is the only thing unusual about them.
`read_lines(src)` -- the community extension duckeye already installs and uses for its
`-f lines` mode -- yields `line_number, content, byte_offset, file_path`, so `#L2-L5` is
`WHERE line_number BETWEEN 2 AND 5`. It returns TEXT LINES, not duck_blocks, unlike every
other resolver here.

For grounding that is fine, and worth being clear about why: C1 checks that a span occurs
verbatim in the text of the thing cited, and the concatenated content of lines 2-5 is
exactly that text. A consumer that needs blocks rather than text should cite `#b<range>`
instead. Neither panduck nor the duck_block vocabulary is involved -- lines are addressed
beside the block model, not through it, which is why searching panduck for line support
finds nothing.

The STRUCTURED form is canonical: it is what `ground_answer._same_page` compares, and two
locators are the same page when their shared fields agree. The string form is a rendering
for citations, CLIs and MCP output, and it must round-trip.
"""
import re
from urllib.parse import unquote

# fragment kind -> the calls that re-read it, tried in order. None = not resolvable yet.
RESOLVERS = {
    "page":    ("read_pdf_blocks({doc}, pages := '{v}')",),
    "blocks":  ("duck_blocks_slice(panduck_read_blocks({doc}), {a}, {b})",),
    "section": ("doc_section({doc}, '{v}')", "doc_container({doc}, '{v}')"),
    "lines":   ("read_lines({doc}) WHERE line_number BETWEEN {a} AND {b}",),
}
WHOLE = "panduck_read_blocks({doc})"

# A named fragment can only resolve if the id survived into the blocks, and THAT depends
# on the installed webbed, not on panduck. On the published build (093856b) only div,
# section/article and headings keep their id, so `<ul id="steps">` is unaddressable;
# webbed 60318d8 widens it to every block behind a capture_attributes parameter, and is
# unreleased. So a `#name` locator is resolvable in principle and may still find nothing
# on a given install -- reported here rather than discovered at query time.
ID_CAPTURE = ("published webbed 093856b captures id on div, section/article and headings "
              "only; 60318d8 widens it to every block but is unreleased")

_PAGE = re.compile(r"^p\.(\d+)(?:-(\d+))?$")
_LINES = re.compile(r"^L(\d+)(?:-L?(\d+))?$")
# Namespaced, unlike pages and lines: see the docstring. The duck is an alias.
DUCK = "\U0001F986"
_BLOCKS = re.compile(r"^(?:duck-blocks|" + DUCK + r"):(\d+)(?:-(\d+))?$")


def parse(text):
    """'foo.html#methods' -> {'doc': 'foo.html', 'section': 'methods'}

    A bare document with no fragment is a legal locator; it names the whole document.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("locator must be a non-empty string")
    doc, _, frag = _split(text)
    loc = {"doc": doc}
    if not frag:
        return loc
    m = _PAGE.match(frag)
    if m:
        loc["page"] = int(m.group(1))
        if m.group(2):
            loc["page_to"] = int(m.group(2))
        return loc
    m = _LINES.match(frag)
    if m:
        loc["lines"] = int(m.group(1))
        if m.group(2):
            loc["lines_to"] = int(m.group(2))
        return loc
    m = _BLOCKS.match(frag)
    if m:
        loc["blocks"] = int(m.group(1))
        if m.group(2):
            loc["blocks_to"] = int(m.group(2))
        return loc
    loc["section"] = frag          # anything else is a heading id or its text
    return loc


def _split(text):
    """Split doc from fragment on the LAST '#'.

    Last, not first: a zim:// entry path or an http URL may legitimately contain '#',
    and the fragment we mean is the one we appended.
    """
    i = text.rfind("#")
    if i <= 0:
        return text, "", ""
    # PERCENT-DECODE the fragment. A fragment is percent-encoded on the wire, so
    # `#%F0%9F%A6%86:12-40` and `#\U0001F986:12-40` are the SAME locator and must parse
    # to the same structure -- otherwise two records citing one passage compare unequal,
    # which surfaces only when someone deduplicates. Decoding also normalises `%20` in a
    # section id back to the space the author wrote.
    return text[:i], "#", unquote(text[i + 1:])


def render(loc, duck=False):
    """The inverse of parse(). Round-trips.

    `duck=True` renders a block range as `#🦆:12-40` instead of `#duck-blocks:12-40`.
    Both parse; the readable one is canonical so output stays greppable by default.
    """
    doc = loc.get("doc")
    if not doc:
        raise ValueError("locator has no doc")
    # The end of a range repeats the prefix for lines and blocks (#L2-L5, #b12-b40)
    # but not for pages (#p.3-7), which is how each is conventionally written. The
    # parser accepts both spellings; render emits the canonical one so it round-trips.
    for key, prefix, repeat in (("page", "p.", False), ("lines", "L", True),
                                ("blocks", "duck-blocks:", False)):
        if key in loc:
            if key == "blocks" and duck:
                prefix = DUCK + ":"
            end = loc.get(key + "_to")
            span = f"{loc[key]}"
            if end is not None:
                span += "-" + (prefix if repeat else "") + f"{end}"
            return f"{doc}#{prefix}{span}"
    if loc.get("section"):
        return f"{doc}#{loc['section']}"
    return doc


def kind(loc):
    """Which positional form this locator uses, or None for a whole document."""
    for k in ("page", "lines", "blocks", "section"):
        if k in loc and loc[k] is not None:
            return k
    return None


def resolvable(loc):
    """Can anything in the stack read this back today?"""
    k = kind(loc)
    return True if k is None else RESOLVERS.get(k) is not None


def _q(doc):
    """SQL-quote the document path."""
    return "'" + str(doc).replace("'", "''") + "'"


def how_to_read(loc):
    """The calls that re-read this locator, in order, or None if nothing does yet.

    A list rather than one call because a named fragment has two resolutions: try
    doc_section, then doc_container. Callers should take the first that returns blocks.

    THE DOCUMENT IS SUBSTITUTED IN, FRAGMENT STRIPPED, and that is not cosmetic. The
    readers below take a document and nothing else: duckeye treats everything after
    `<archive>.zim/` as the entry name VERBATIM and does no fragment parsing, by
    design -- resolving the fragment is this module's job. So handing a reader the whole
    locator asks it for an entry literally named `C#Sharp#overview`, which does not
    exist. Emitting the stripped doc means a caller cannot make that mistake by pasting
    what it was given.
    """
    k = kind(loc)
    doc = _q(loc.get("doc"))
    if k is None:
        return [WHOLE.format(doc=doc)]
    tmpls = RESOLVERS.get(k)
    if tmpls is None:
        return None
    if k in ("blocks", "lines"):
        # Range forms: a single value is a degenerate range (#L7 -> 7 AND 7).
        return [t.format(doc=doc, a=loc[k], b=loc.get(k + "_to", loc[k])) for t in tmpls]
    end = loc.get(k + "_to")
    v = f"{loc[k]}" + (f"-{end}" if end is not None else "")
    return [t.format(doc=doc, v=v) for t in tmpls]
