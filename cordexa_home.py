"""Locate the cordexa checkout whose `reference/` this library loads from.

Both layers over cordexa need this path — `ground_answer` for C1 (`c1_span.py`)
and `c3` for C3 (`c3_support.py`) — and each used to carry its own copy of the
default. The copies are why the bug was invisible: fixing one left the other
resolving to a directory that does not exist, and the failure surfaces as a
`FileNotFoundError` at import time, which reads as a broken test module rather
than a wrong path.

The default has to cope with two checkout shapes. A plain clone puts
`reference/` directly under `~/Projects/cordexa`; a git-wt checkout puts the
working tree in `~/Projects/cordexa/main` beside `trees/`. Probing for the file
we actually need distinguishes them without guessing, so neither shape needs
`CORDEXA_HOME` set just to run the tests.
"""
import os

CANDIDATES = ("~/Projects/cordexa/main", "~/Projects/cordexa")


def find(marker=os.path.join("reference", "c1_span.py")):
    """-> the cordexa root. CORDEXA_HOME wins outright if it is set.

    `marker` is a file that must exist under the root for it to count. It
    defaults to a reference module rather than to `reference/` itself so an
    empty or half-populated directory does not win over a real checkout.

    When nothing matches, returns the last candidate unchanged: the caller then
    fails on the missing file with a path in the message, which says more than
    an exception raised here with no attempted path in it.
    """
    env = os.environ.get("CORDEXA_HOME")
    roots = [env] if env else list(CANDIDATES)
    for r in roots:
        p = os.path.expanduser(r)
        if os.path.exists(os.path.join(p, marker)):
            return p
    return os.path.expanduser(roots[-1])
