"""Machine-local settings: the environment first, then a local file, then the default.

The data paths (Vortex shards, federation text DBs, ZIM archive directories) differ per
host, and the repo is public, so no host's paths are written into the code. A host names
them in `~/.config/tiibrarian/paths.env` (or the file `$TIIBRARIAN_CONFIG` names):

    TIIBRARIAN_VORTEX_SHARDS=/path/survivorlibrary_vectors.vortex:/path/zim_vectors.vortex
    TIIBRARIAN_ZIM_SHARDS=/path/zim_vectors.vortex
    TIIBRARIAN_SL_DB=/path/survivorlibrary.duckdb
    TIIBRARIAN_ZIM_DIRS=/path/zim

A FILE rather than shell exports, because several callers never see a login shell: a
systemd unit, another repo's script, a Slurm job. One file per host serves all of them.
`KEY=VALUE`, absolute paths, no quotes and no expansion -- the same lines systemd's
`EnvironmentFile=` and `set -a; . file` accept, so the file can be shared with both.

Nothing here raises. An unset path is `""` (or `[]`), and the module that USES it raises
with `require`, so importing never fails on a host that has no file.
"""
import os

FILE = os.path.expanduser(os.environ.get("TIIBRARIAN_CONFIG", "~/.config/tiibrarian/paths.env"))


def _read(path):
    out = {}
    try:
        f = open(path)
    except FileNotFoundError:
        return out
    with f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def setting(name, default="", path=None):
    """$name if set and non-empty, else the file's value, else `default`."""
    return os.environ.get(name) or _read(path or FILE).get(name) or default


def paths(name, default="", path=None):
    """A colon-delimited setting as a list. Empty entries dropped, so unset is `[]`."""
    return [p for p in setting(name, default, path).split(":") if p]


def require(value, name):
    """`value`, or raise naming the setting and where to make it."""
    if not value:
        raise RuntimeError(
            f"{name} is not set. Set it in the environment, or as a {name}=... line in "
            f"{FILE} (the file $TIIBRARIAN_CONFIG names). This host's data paths are not "
            f"in the code, by design.")
    return value
