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
    #b12-b40                 block range         -> duck_blocks_slice(blocks, 12, 40)
    #L2-L5                   line range          -> NOTHING RESOLVES THIS YET
    #methods                 section id or text  -> doc_section(src, 'methods')

`#b…` is the format-agnostic one: every duck_block carries `element_order` whatever the
reader was, so a block range addresses HTML, Markdown, DOCX, EPUB and PDF identically.
Pages only exist in paginated formats; sections only where there are headings; lines only
in text-ish sources. Prefer the most specific form the format actually supports, and fall
back to `#b…`, which always exists.

WHY LINES ARE LISTED BUT NOT SUPPORTED: `#L2-L5` is the obvious spelling for Markdown and
source files, and it is what a person would write. Nothing in the stack addresses lines
today -- panduck has no line-range reader and duck_blocks carry no line numbers -- so
`resolvable()` returns False for it. It parses and round-trips so a corpus can record one,
and the day a reader gains line addressing only `RESOLVERS` changes.

The STRUCTURED form is canonical: it is what `ground_answer._same_page` compares, and two
locators are the same page when their shared fields agree. The string form is a rendering
for citations, CLIs and MCP output, and it must round-trip.
"""
import re

# fragment kind -> how it is re-read. None means "not resolvable yet".
RESOLVERS = {
    "page":    "read_pdf_blocks(src, pages := '{v}')",
    "blocks":  "duck_blocks_slice(panduck_read_blocks(src), {a}, {b})",
    "section": "doc_section(src, '{v}')",
    "lines":   None,
}

_PAGE = re.compile(r"^p\.(\d+)(?:-(\d+))?$")
_LINES = re.compile(r"^L(\d+)(?:-L?(\d+))?$")
_BLOCKS = re.compile(r"^b(\d+)(?:-b?(\d+))?$")


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
    return text[:i], "#", text[i + 1:]


def render(loc):
    """The inverse of parse(). Round-trips."""
    doc = loc.get("doc")
    if not doc:
        raise ValueError("locator has no doc")
    # The end of a range repeats the prefix for lines and blocks (#L2-L5, #b12-b40)
    # but not for pages (#p.3-7), which is how each is conventionally written. The
    # parser accepts both spellings; render emits the canonical one so it round-trips.
    for key, prefix, repeat in (("page", "p.", False), ("lines", "L", True),
                                ("blocks", "b", True)):
        if key in loc:
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


def how_to_read(loc):
    """The call that re-reads this locator, or None if nothing does yet."""
    k = kind(loc)
    if k is None:
        return "panduck_read_blocks(src)"
    tmpl = RESOLVERS.get(k)
    if tmpl is None:
        return None
    if k in ("blocks",):
        return tmpl.format(a=loc[k], b=loc.get(k + "_to", loc[k]))
    end = loc.get(k + "_to")
    v = f"{loc[k]}" + (f"-{end}" if end is not None else "")
    return tmpl.format(v=v)
