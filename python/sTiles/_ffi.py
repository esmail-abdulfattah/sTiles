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
import json
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
        base = "libstiles-macos-apple-arm64" if arch == "arm64" \
            else "libstiles-macos-intel-x86_64"
    elif sys.platform.startswith("linux"):
        base = f"libstiles-linux-{arch}"
    elif sys.platform.startswith("win"):
        base = f"libstiles-windows-{arch}"
    else:
        base = f"libstiles-{sys.platform}-{arch}"
    # Opt-in build variants (see BUILDS.md in the source repository), e.g.
    #   STILES_VARIANT=v3-mkl        (Linux x86_64: newest GCC, needs glibc 2.38+)
    #   STILES_VARIANT=armv82-armpl  (Linux arm64: newest GCC + ARMPL)
    # The default remains the most compatible build for the platform.
    variant = os.environ.get("STILES_VARIANT", "").strip()
    return f"{base}-{variant}" if variant else base


def _check_cpu_supported() -> None:
    """Fail with a clear message on CPUs the prebuilt library cannot run on.

    The x86_64 builds are compiled for AVX2 (Intel Haswell 2013+ / AMD
    Excavator+). Without this check an unsupported machine downloads the
    library fine and then dies with an uninformative ``Illegal instruction``
    at the first factorization.
    """
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        return
    have_avx2 = True
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as fh:
                have_avx2 = " avx2" in fh.read()
        elif sys.platform.startswith("win"):
            import ctypes
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
            have_avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
        # macOS Intel: every machine Apple still supports has AVX2.
    except Exception:
        return  # never block loading on a failed detection
    if not have_avx2:
        raise RuntimeError(
            "the prebuilt sTiles library requires a CPU with AVX2 "
            "(Intel Haswell 2013+ or AMD Excavator+); this machine does not "
            "report it. Build sTiles from source for this CPU instead."
        )


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

    # Cache from a previous release download, NEWEST release first. The flat
    # (pre-versioning) path is deliberately NOT offered here: it is the layout
    # that pinned users to whichever solver they first downloaded, so it is
    # only ever used as an offline fallback inside _download_from_release().
    cands.extend(sorted(_cache_dir().glob(f"*/{ci}/{fname}"), reverse=True))

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


def clear_cache() -> int:
    """Delete every cached solver. Returns how many were removed.

    The compiled solver is downloaded separately from this package and cached,
    so reinstalling the package does NOT replace it. Use this to start fresh.
    """
    root = _cache_dir()
    n = 0
    if root.is_dir():
        for entry in root.iterdir():
            shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
            n += 1
    return n


def update() -> Path:
    """Download the current released solver, replacing any cached copy.

    Restart Python afterwards: the solver is loaded once per process.
    """
    clear_cache()
    lib = _download_from_release(force=True)
    if lib is None:
        raise RuntimeError(
            "could not download a solver; check the network, or set STILES_LIB "
            "to a local libstiles"
        )
    return lib


def _latest_tag() -> str | None:
    """Current release tag, so the cache can be keyed by it. None when offline."""
    if os.environ.get("STILES_RELEASE_TAG"):
        return os.environ["STILES_RELEASE_TAG"]
    try:
        url = f"https://api.github.com/repos/{_RELEASE_REPO}/releases/latest"
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            return json.load(resp).get("tag_name") or None
    except Exception:  # noqa: BLE001 - offline is normal, not an error
        return None


def _cached_libs(ci: str, fname: str) -> list[Path]:
    """Every solver already cached, newest release first (the offline path)."""
    root = _cache_dir()
    hits = sorted(root.glob(f"*/{ci}/{fname}"), reverse=True)
    flat = root / ci / fname          # pre-versioning layout
    if flat.is_file():
        hits.append(flat)
    return [p for p in hits if p.is_file()]


def _download_from_release(force: bool = False) -> Path | None:
    """Download the matching libstiles into the cache; return its path or None."""
    if os.environ.get("STILES_NO_DOWNLOAD"):
        return None
    ci = _ci_folder()
    fname = _lib_filename()

    # Cache keyed by RELEASE, not just platform. Keyed by platform alone (the
    # original layout) the first download became permanent: later releases were
    # never fetched, reinstalling the package changed nothing, and users kept a
    # solver months old -- including one predating the fix for a bug they were
    # hitting. Same defect, same fix, as the R package.
    tag = _latest_tag()
    if tag is None:                                  # offline: use what we have
        have = _cached_libs(ci, fname)
        return have[0] if have else None
    dest = _cache_dir() / tag / ci
    lib_path = dest / fname
    if lib_path.is_file() and not force:
        return lib_path

    base = os.environ.get(
        "STILES_RELEASE_BASE_URL",
        f"https://github.com/{_RELEASE_REPO}/releases/download/{tag}",
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
                    # Everything shipped under lib/: the library itself,
                    # plus, on platforms that aren't fully self-contained
                    # (macOS Intel, Windows), the sibling runtime
                    # .dylib/.so/.dll it loads via a loader-relative path.
                    # Extracting only the exact library filename left those
                    # siblings behind and broke the load on non-self-contained
                    # builds.
                    if not member.startswith("lib/") or member.endswith("/"):
                        continue
                    bn = os.path.basename(member)
                    with zf.open(member) as src, open(dest / bn, "wb") as out:
                        shutil.copyfileobj(src, out)
        finally:
            os.unlink(tmp_zip)
        # Superseded solvers are ~20 MB each and serve nobody once a newer one
        # loads; drop them, including any pre-versioning copy.
        for old_dir in _cache_dir().iterdir():
            if old_dir.is_dir() and old_dir.name not in (tag,):
                if old_dir.name == ci or old_dir.name[:1].isdigit() or old_dir.name.startswith("v"):
                    shutil.rmtree(old_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - any failure -> fall through to error
        sys.stderr.write(f"sTiles: release download failed ({exc})\n")
        return None
    return lib_path if lib_path.is_file() else None


def _load() -> tuple[ctypes.CDLL, str]:
    _check_cpu_supported()
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
