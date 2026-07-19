"""
Low-level ctypes binding to libstiles.

This module does exactly three things:

  1. Locate the shared library (``libstiles.so`` on Linux, ``libstiles.dylib``
     on macOS) for the current platform.
  2. ``ctypes.CDLL`` it.
  3. Declare the ``argtypes`` / ``restype`` for every ``sTiles_*`` entry point
     the high-level wrapper uses.

Everything user-facing lives in :mod:`sTiles.core`; this file is the raw
FFI surface and has no NumPy/SciPy dependency.

Library search order (first hit wins)
-------------------------------------
  1. ``$STILES_LIB``            -- full path to the shared object.
  2. ``$STILES_LIB_DIR``        -- directory containing it.
  3. ``$STILES_BINARIES_DIR``   -- a CI-artifact tree with
                                   ``libstiles-<plat>/lib/libstiles.{so,dylib}``.
  4. a ``binaries/`` (or ``bindings/binaries/``) directory in any ancestor,
     using the same CI-artifact layout -- so unzipped GitHub Actions artifacts
     dropped in ``bindings/binaries/`` are found with zero configuration.
  5. bundled ``_libs/<plat>/``  -- shipped inside the wheel / R package by CI.
  6. repo dev fallback          -- ``lib/libstiles.{so,dylib}`` in an ancestor.
  7. GitHub Release download     -- the matching platform library is fetched from
                                   the project's Release assets and cached (this
                                   is what makes ``pip install`` work). Disable
                                   with ``$STILES_NO_DOWNLOAD=1``.

The library built for Linux embeds MKL/SCOTCH/METIS statically and localizes
their symbols, so a plain ``CDLL`` is safe even inside a process that already
loaded its own MKL (e.g. R, or MKL-backed NumPy/SciPy).  The macOS build links Homebrew OpenBLAS +
LAPACK, so those must be discoverable at load time on a Mac.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_double,
    c_int,
    c_longlong,
    c_void_p,
)
from pathlib import Path

__all__ = ["lib", "library_path", "c_int_p", "c_double_p", "c_bool_p"]

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)
c_bool_p = POINTER(c_bool)

# Suppress the one-time ASCII banner libstiles prints on first use.  Set before
# the library is loaded so its getenv() sees it; overridable by exporting
# STILES_NO_BANNER=0 for users who want the banner.
os.environ.setdefault("STILES_NO_BANNER", "1")


def _platform_tag() -> str:
    """Return the ``<os>-<arch>`` sub-directory name used for bundled libs."""
    machine = platform.machine().lower()
    arch = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(machine, machine)
    if sys.platform == "darwin":
        return f"macos-{arch}"
    if sys.platform.startswith("linux"):
        return f"linux-{arch}"
    return f"{sys.platform}-{arch}"


def _ci_folder() -> str:
    """Name of the CI build-artifact directory for this platform.

    The GitHub Actions ``build`` workflow uploads one directory per target as
    ``libstiles-<name>/lib/libstiles.{so,dylib}`` -- these names differ from the
    bundle ``<os>-<arch>`` tag (e.g. ``libstiles-macos-apple-arm64``).
    """
    machine = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64",
            "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    if sys.platform == "darwin":
        return "libstiles-macos-apple-arm64" if arch == "arm64" \
            else "libstiles-macos-intel-x86_64"
    if sys.platform.startswith("linux"):
        return f"libstiles-linux-{arch}"
    if sys.platform.startswith("win"):
        return f"libstiles-windows-{arch}"
    return f"libstiles-{sys.platform}-{arch}"


def _lib_filename() -> str:
    if sys.platform == "darwin":
        return "libstiles.dylib"
    if sys.platform.startswith("win"):
        return "libstiles.dll"
    return "libstiles.so"


def _candidate_paths() -> list[Path]:
    fname = _lib_filename()
    ci = _ci_folder()
    cands: list[Path] = []

    env_lib = os.environ.get("STILES_LIB")
    if env_lib:
        cands.append(Path(env_lib))

    env_dir = os.environ.get("STILES_LIB_DIR")
    if env_dir:
        cands.append(Path(env_dir) / fname)

    here = Path(__file__).resolve().parent

    # CI binaries tree: <root>/libstiles-<ci>/lib/libstiles.{so,dylib}.
    # Explicit root, then any `binaries/` or `bindings/binaries/` above us.
    env_bin = os.environ.get("STILES_BINARIES_DIR")
    if env_bin:
        cands.append(Path(env_bin) / ci / "lib" / fname)
    for parent in [here, *here.parents]:
        cands.append(parent / "binaries" / ci / "lib" / fname)
        cands.append(parent / "bindings" / "binaries" / ci / "lib" / fname)

    # Bundled: sTiles/_libs/<plat>/libstiles.{so,dylib} and a flat fallback.
    cands.append(here / "_libs" / _platform_tag() / fname)
    cands.append(here / "_libs" / fname)

    # Cache dir from a previous release download (see _download_from_release).
    cands.append(_cache_dir() / ci / fname)

    # Repo dev fallback: search ancestors for lib/libstiles.{so,dylib}.
    for parent in [here, *here.parents]:
        cands.append(parent / "lib" / fname)

    return cands


# ---------------------------------------------------------------------------
# Fetch the matching prebuilt libstiles from the GitHub Release.
#
# When pip-installed there is no binary in the tree, so on first use we download
# the platform library from the project's Release assets and cache it under the
# user cache dir.  The Linux/macOS builds are self-contained (BLAS embedded), so
# the cached file loads with no extra system packages.
#
# Overrides:
#   $STILES_NO_DOWNLOAD=1      -- never hit the network (raise instead).
#   $STILES_RELEASE_REPO       -- "owner/repo" hosting the Release (default below).
#   $STILES_RELEASE_BASE_URL   -- full base URL for the assets (bypasses the repo).
#   $STILES_CACHE_DIR          -- where to cache the downloaded library.
# ---------------------------------------------------------------------------
_RELEASE_REPO = os.environ.get("STILES_RELEASE_REPO", "esmail-abdulfattah/sTiles")


def _cache_dir() -> Path:
    env = os.environ.get("STILES_CACHE_DIR")
    if env:
        return Path(env)
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "sTiles"
    base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(base) / "sTiles"


def _download_from_release() -> Path | None:
    """Download the matching libstiles into the cache; return its path or None."""
    if os.environ.get("STILES_NO_DOWNLOAD"):
        return None
    ci = _ci_folder()
    fname = _lib_filename()
    dest = _cache_dir() / ci
    lib_path = dest / fname
    if lib_path.is_file():
        return lib_path  # already downloaded on a previous run

    base = os.environ.get(
        "STILES_RELEASE_BASE_URL",
        f"https://github.com/{_RELEASE_REPO}/releases/latest/download",
    )
    url = f"{base}/{ci}.zip"
    try:
        dest.mkdir(parents=True, exist_ok=True)
        sys.stderr.write(f"sTiles: fetching libstiles for {ci} from {url}\n")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with urllib.request.urlopen(url) as resp:  # noqa: S310
                shutil.copyfileobj(resp, tmp)
            tmp_zip = tmp.name
        try:
            with zipfile.ZipFile(tmp_zip) as zf:
                for member in zf.namelist():
                    bn = os.path.basename(member)
                    # The library, plus (Windows) the sibling runtime DLLs.
                    want = bn == fname or (
                        sys.platform.startswith("win") and bn.lower().endswith(".dll")
                    )
                    if want:
                        with zf.open(member) as src, open(dest / bn, "wb") as out:
                            shutil.copyfileobj(src, out)
        finally:
            os.unlink(tmp_zip)
    except Exception as exc:  # noqa: BLE001 - any failure -> fall through to error
        sys.stderr.write(f"sTiles: release download failed ({exc})\n")
        return None
    return lib_path if lib_path.is_file() else None


def _load() -> tuple[ctypes.CDLL, str]:
    tried: list[str] = []
    for path in _candidate_paths():
        tried.append(str(path))
        if path.is_file():
            try:
                return ctypes.CDLL(str(path)), str(path)
            except OSError as exc:  # pragma: no cover - surfaced below
                tried[-1] += f"  (load failed: {exc})"

    # Nothing local: fetch the prebuilt library from the GitHub Release.
    downloaded = _download_from_release()
    if downloaded is not None:
        tried.append(str(downloaded))
        try:
            return ctypes.CDLL(str(downloaded)), str(downloaded)
        except OSError as exc:
            tried[-1] += f"  (load failed: {exc})"

    # Last resort: let the loader resolve a bare SONAME via LD_LIBRARY_PATH.
    try:
        return ctypes.CDLL(_lib_filename()), _lib_filename()
    except OSError:
        pass

    raise OSError(
        "Could not locate libstiles for this platform.\n"
        "The automatic download from the GitHub Release failed or was disabled.\n"
        "Set $STILES_LIB to the shared object, point $STILES_BINARIES_DIR at a\n"
        f"CI-artifact tree ({_ci_folder()}/lib/{_lib_filename()}), or drop it in\n"
        f"sTiles/_libs/{_platform_tag()}/{_lib_filename()}.\nSearched:\n  "
        + "\n  ".join(tried)
    )


lib, library_path = _load()


# ---------------------------------------------------------------------------
# Prototype declarations.
#
# The handle is an opaque ``void*``; every lifecycle call takes ``void**`` which
# on the Python side is ``byref(c_void_p)``.  We declare those params as
# ``c_void_p`` (a pointer-to-pointer is still just an address) and always pass
# ``ctypes.byref(handle)`` at the call site.
# ---------------------------------------------------------------------------
def _decl(name, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


# Version / logging ---------------------------------------------------------
_decl("sTiles_get_version", c_char_p, [])
_decl("sTiles_set_log_level", None, [c_int])
_decl("sTiles_expert_user", None, [])

# Global configuration ------------------------------------------------------
_decl("sTiles_set_tile_size", None, [c_int])
_decl("sTiles_return_tile_size", c_int, [])
_decl("sTiles_get_auto_tile_size", c_int, [])
_decl("sTiles_set_tile_type_mode", None, [c_int])
_decl("sTiles_set_ordering_mode", None, [c_int])
_decl("sTiles_force_ND", None, [c_int])

# Lifecycle -----------------------------------------------------------------
# int sTiles_create(void**, int num_groups, const int* calls_per_group,
#                   const int* cores_per_group, const int* chol_type,
#                   const bool* get_inverse)
_decl("sTiles_create", c_int,
      [c_void_p, c_int, c_int_p, c_int_p, c_int_p, c_bool_p])
# int sTiles_assign_graph_one_call(int g, int c, void**, int n, int nnz,
#                                  int* row, int* col)
_decl("sTiles_assign_graph_one_call", c_int,
      [c_int, c_int, c_void_p, c_int, c_int, c_int_p, c_int_p])
_decl("sTiles_init_group", c_int, [c_int, c_void_p])
_decl("sTiles_assign_values", c_int, [c_int, c_int, c_void_p, c_double_p])
_decl("sTiles_bind", c_int, [c_int, c_int, c_void_p])
_decl("sTiles_unbind", c_int, [c_int, c_int, c_void_p])
_decl("sTiles_chol", c_int, [c_int, c_int, c_void_p])
_decl("sTiles_selinv", c_int, [c_int, c_int, c_void_p])
_decl("sTiles_freeGroup", None, [c_int])
_decl("sTiles_quit", None, [])

# Result accessors ----------------------------------------------------------
_decl("sTiles_get_logdet", c_double, [c_int, c_int, c_void_p])
_decl("sTiles_get_nnz_factor", c_longlong, [c_int, c_int, c_void_p])
_decl("sTiles_get_selinv_elm", c_double, [c_int, c_int, c_int, c_int, c_void_p])
_decl("sTiles_get_chol_elm", c_double, [c_int, c_int, c_int, c_int, c_void_p])
# double* sTiles_get_selinv_row(int g, int c, int node, int* neighbors,
#                               int size, void**)
_decl("sTiles_get_selinv_row", c_double_p,
      [c_int, c_int, c_int, c_int_p, c_int, c_void_p])
_decl("sTiles_clear_selinv", c_int, [c_int, c_int, c_void_p])
_decl("sTiles_get_chol_timing", c_double, [c_int, c_int, c_void_p])
_decl("sTiles_get_selinv_timing", c_double, [c_int, c_int, c_void_p])

# Permutation ---------------------------------------------------------------
# int sTiles_get_logical_element_perm(int g, int c, void**, int* out_perm)
_decl("sTiles_get_logical_element_perm", c_int, [c_int, c_int, c_void_p, c_int_p])

# Solvers (B is column-major, original order, overwritten in place) ----------
_decl("sTiles_solve_LLT", c_int, [c_int, c_int, c_void_p, c_double_p, c_int])
_decl("sTiles_solve_L", c_int, [c_int, c_int, c_void_p, c_double_p, c_int])
_decl("sTiles_solve_LT", c_int, [c_int, c_int, c_void_p, c_double_p, c_int])

# Memory estimate (static, no handle) --------------------------------------
_decl("sTiles_estimate_memory", c_double,
      [c_int, c_int, c_int, c_int, c_int, c_int])
