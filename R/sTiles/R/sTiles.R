## sTiles -- R interface to the sTiles sparse Cholesky / selected-inverse
## framework.  See ?sTiles for the entry point.

#' @name sTiles-package
#' @title Tile-Based Sparse Cholesky Factorization and Selected Inverse
#' @description
#' An interface to the sTiles framework: sparse Cholesky factorization,
#' log-determinants, selected inverse (marginal variances) and triangular
#' solves, organised so that the symbolic work is paid once per sparsity
#' pattern and reused across every set of values sharing it.
#'
#' Start at [sTiles()] for a one-shot factorization, or at [sTiles_analyze()]
#' plus [sTiles_factorize()] when many matrices share one pattern.
#'
#' @section The solver library:
#' The numerical work happens in libstiles, a separate component under its own
#' license terms that this package loads at run time (see the NOTICE file).
#' [sTiles_available()] reports whether a copy is present,
#' [sTiles_install_library()] fetches one, and `STILES_LIB` points the package
#' at a copy you already have.
#'
#' @keywords internal
NULL

# Package-private state: the located libstiles path, the glue DLLInfo, and a
# cache of resolved native symbols.
.sTiles <- new.env(parent = emptyenv())

# Tile factorization regimes (mirror stiles.h tile_type_mode values).
.sTiles_modes <- c(dense = 0L, semisparse = 1L, semi = 1L,
                   sparse = 2L, auto = 3L)

# ---------------------------------------------------------------------------
# Library resolution -- mirrors the Python binding's search order.
# ---------------------------------------------------------------------------
.sTiles_lib_filename <- function() {
    switch(Sys.info()[["sysname"]],
           Darwin  = "libstiles.dylib",
           Windows = "libstiles.dll",
           "libstiles.so")
}

.sTiles_platform_tag <- function() {
    sysname <- Sys.info()[["sysname"]]
    machine <- Sys.info()[["machine"]]
    arch <- switch(machine,
                   "x86_64" = "x86_64", "amd64" = "x86_64",
                   "arm64" = "arm64", "aarch64" = "arm64", machine)
    os <- switch(sysname, Darwin = "macos", Windows = "windows", "linux")
    paste0(os, "-", arch)
}

# CPU feature list from the kernel, empty when it cannot be read.
.sTiles_arm_cpu_flags <- function() {
    if (Sys.info()[["sysname"]] != "Linux" || !file.exists("/proc/cpuinfo")) return(character(0))
    ln <- tryCatch(readLines("/proc/cpuinfo", warn = FALSE), error = function(e) character(0))
    f  <- grep("^Features", ln, value = TRUE)
    if (!length(f)) return(character(0))
    strsplit(trimws(sub("^Features\\s*:", "", f[1])), "\\s+")[[1]]
}

# Best-fitting build variant for this CPU, or "" for the portable default.
# Only Linux arm64 is auto-selected, and only where the gain is a real ISA
# difference the default build cannot use:
#   sve2                -> armv9-sve2-armpl  (Grace, Graviton4, N2, Cortex-X925)
#   LSE atomics + RDMA  -> armv82-armpl      (Graviton2+, Ampere Altra)
#   otherwise           -> the baseline armv8 asset
# NOT auto-selected: x86_64 (the v3 asset is -march=x86-64-v3, no faster than
# the default haswell build, and it raises the glibc floor to 2.38) and macOS
# (the default arm64 build is already -mcpu=apple-m1; the -gcc-armpl mirrors
# exist for linking into GCC programs, not for speed).
# The selected variants embed ARM Performance Libraries: ~45 MB against ~5 MB
# for the baseline. Set STILES_VARIANT=none to decline it.
.sTiles_auto_variant <- function() {
    if (Sys.info()[["sysname"]] != "Linux") return("")
    machine <- Sys.info()[["machine"]]
    if (!machine %in% c("aarch64", "arm64")) return("")
    flags <- .sTiles_arm_cpu_flags()
    if ("sve2" %in% flags) return("armv9-sve2-armpl")
    ## -march=armv8.2-a lets the compiler emit LSE atomics (v8.1) and SQRDMLAH
    ## (RDMA, v8.1). Require both rather than trusting a marketing name: a core
    ## without them SIGILLs on the first tiled update.
    if (all(c("atomics", "asimdrdm") %in% flags)) return("armv82-armpl")
    ""
}

# Name of the CI build-artifact directory for this platform, e.g.
# "libstiles-linux-x86_64" or "libstiles-macos-apple-arm64".
.sTiles_ci_folder <- function() {
    sysname <- Sys.info()[["sysname"]]
    machine <- Sys.info()[["machine"]]
    arch <- switch(machine,
                   "x86_64" = "x86_64", "amd64" = "x86_64",
                   "arm64" = "arm64", "aarch64" = "arm64", machine)
    base <- if (sysname == "Darwin") {
        if (arch == "arm64") "libstiles-macos-apple-arm64"
        else "libstiles-macos-intel-x86_64"
    } else if (sysname == "Windows") {
        paste0("libstiles-windows-", arch)
    } else {
        paste0("libstiles-linux-", arch)
    }
    ## Build variant. An explicit STILES_VARIANT always wins, so any published
    ## asset can be pinned (v3-mkl, armv82-armpl, armv9-sve2-armpl, ...);
    ## STILES_VARIANT=none forces the portable default. With nothing set the
    ## CPU picks (.sTiles_auto_variant), and .sTiles_ci_candidates falls back
    ## to the default asset when the chosen one is not in the release.
    variant <- trimws(Sys.getenv("STILES_VARIANT", ""))
    if (tolower(variant) %in% c("none", "default", "base")) return(base)
    if (!nzchar(variant)) variant <- .sTiles_auto_variant()
    if (nzchar(variant)) paste0(base, "-", variant) else base
}

# Asset names to try, best first, always ending at the portable default.
.sTiles_ci_candidates <- function() {
    sel <- .sTiles_ci_folder()
    old <- Sys.getenv("STILES_VARIANT", NA_character_)
    Sys.setenv(STILES_VARIANT = "none")
    on.exit(if (is.na(old)) Sys.unsetenv("STILES_VARIANT") else Sys.setenv(STILES_VARIANT = old))
    unique(c(sel, .sTiles_ci_folder()))
}

# Walk up from `start`, collecting CI-artifact candidates
#   <ancestor>/binaries/<ci>/lib/<fname>  and  <ancestor>/bindings/binaries/...
.sTiles_binaries_candidates <- function(start, ci, fname) {
    out <- character(0)
    here <- start
    for (i in seq_len(12)) {
        out <- c(out,
                 file.path(here, "binaries", ci, "lib", fname),
                 file.path(here, "bindings", "binaries", ci, "lib", fname))
        parent <- dirname(here)
        if (parent == here) break
        here <- parent
    }
    out
}

## The prebuilt x86_64 libraries are compiled for AVX2 (Intel Haswell
## 2013+ / AMD Excavator+). Refuse loading with a clear message instead of
## letting an old CPU die on "illegal instruction" mid-factorization.
.sTiles_check_cpu <- function() {
    machine <- tolower(Sys.info()[["machine"]])
    if (!machine %in% c("x86_64", "amd64", "x86-64")) return(invisible(TRUE))
    ok <- TRUE
    if (Sys.info()[["sysname"]] == "Linux" && file.exists("/proc/cpuinfo")) {
        ok <- tryCatch(
            any(grepl("\\bavx2\\b", readLines("/proc/cpuinfo", warn = FALSE))),
            error = function(e) TRUE)   # never block on a failed detection
    }
    if (!ok)
        stop("the prebuilt sTiles library requires a CPU with AVX2 ",
             "(Intel Haswell 2013+ or AMD Excavator+); this machine does not ",
             "report it. Build sTiles from source for this CPU instead.",
             call. = FALSE)
    invisible(TRUE)
}

# ---------------------------------------------------------------------------
# The solver library is a separate component and is never fetched behind the
# user's back: sTiles_install_library() puts a copy in the cache below, and
# that call is the only path in this package that reaches the network. The
# Linux/macOS builds are self-contained (BLAS embedded).
# Overrides: STILES_NO_DOWNLOAD, STILES_RELEASE_REPO, STILES_RELEASE_BASE_URL,
# STILES_CACHE_DIR, STILES_RELEASE_TAG, STILES_VARIANT.
# ---------------------------------------------------------------------------
.sTiles_cache_dir <- function() {
    env <- Sys.getenv("STILES_CACHE_DIR", "")
    if (nzchar(env)) return(env)
    tools::R_user_dir("sTiles", which = "cache")
}

# Which release is current? One request to the releases API, so the cache can
# be keyed by TAG. Returns NA offline, and every caller must cope with that.
.sTiles_latest_tag <- function(repo) {
    tryCatch({
        js <- paste(readLines(sprintf("https://api.github.com/repos/%s/releases/latest", repo),
                              warn = FALSE), collapse = "")
        m <- regmatches(js, regexpr('"tag_name"[^"]*"[^"]+"', js))
        if (length(m) == 1L) sub('.*"tag_name"[^"]*"([^"]+)".*', "\\1", m) else NA_character_
    }, error = function(e) NA_character_, warning = function(w) NA_character_)
}

# Any solver already cached, newest release first. The offline fallback, and
# what makes a failed API call harmless rather than fatal.
.sTiles_cached_libs <- function(ci, fname) {
    root <- .sTiles_cache_dir()
    hits <- Sys.glob(file.path(root, "*", ci, fname))
    hits <- c(hits, file.path(root, ci, fname))   # pre-versioning layout
    hits[file.exists(hits)]
}

# Fetch the best asset for this CPU, falling back to the portable default.
# .sTiles_ci_candidates() is best-first and always ends at the default build,
# so a release that does not carry the CPU-specific asset (an older tag, or a
# lane that failed to publish) still installs instead of erroring.
.sTiles_download_from_release <- function(force = FALSE) {
    if (nzchar(Sys.getenv("STILES_NO_DOWNLOAD", ""))) return(NA_character_)
    for (cand in .sTiles_ci_candidates()) {
        got <- .sTiles_download_one(cand, force)
        if (!is.na(got)) return(got)
    }
    NA_character_
}

.sTiles_download_one <- function(ci, force = FALSE) {
    fname <- .sTiles_lib_filename()
    repo  <- Sys.getenv("STILES_RELEASE_REPO", "esmail-abdulfattah/sTiles")
    tag   <- Sys.getenv("STILES_RELEASE_TAG", "")
    if (!nzchar(tag)) tag <- .sTiles_latest_tag(repo)

    # The cache is keyed by RELEASE, not just platform. Keyed by platform
    # alone (the original layout), the first download became permanent: every
    # later release was ignored, reinstalling the package changed nothing, and
    # users silently kept a solver months old -- including one that predated a
    # fix for the bug they were hitting.
    if (!is.na(tag)) {
        dest <- file.path(.sTiles_cache_dir(), tag, ci)
        lib  <- file.path(dest, fname)
        if (file.exists(lib) && !force) return(lib)
    } else {
        # Offline: use whatever is already cached rather than failing.
        have <- .sTiles_cached_libs(ci, fname)
        if (length(have)) return(sort(have, decreasing = TRUE)[1])
        return(NA_character_)
    }

    base <- Sys.getenv("STILES_RELEASE_BASE_URL",
                       sprintf("https://github.com/%s/releases/download/%s", repo, tag))
    url <- sprintf("%s/%s.zip", base, ci)
    ok <- tryCatch({
        dir.create(dest, recursive = TRUE, showWarnings = FALSE)
        tmp <- tempfile(fileext = ".zip")
        message(sprintf("sTiles: fetching libstiles %s for %s", tag, ci))
        utils::download.file(url, tmp, mode = "wb", quiet = TRUE)
        entries <- utils::unzip(tmp, list = TRUE)$Name
        # Everything shipped under lib/: the library itself, plus, on
        # platforms that aren't fully self-contained (macOS Intel, Windows),
        # the sibling runtime .dylib/.so/.dll it loads via a loader-relative
        # path. Extracting only the exact library filename left those
        # siblings behind and broke the load on non-self-contained builds.
        want <- entries[startsWith(entries, "lib/") & !endsWith(entries, "/")]
        # Stage, then move into place. Unzipping straight into `dest` rewrites
        # the .so where it already sits, and when that same file is mapped into
        # this process (force = TRUE over a loaded solver) the mapping turns to
        # garbage underneath it and R dies with an irrecoverable exception.
        # unlink + rename swaps the directory ENTRY instead: the running
        # process keeps the old inode, intact, until it unloads it.
        stage <- file.path(dest, ".stage")
        unlink(stage, recursive = TRUE)
        utils::unzip(tmp, files = want, exdir = stage, junkpaths = TRUE)
        unlink(tmp)
        for (f in list.files(stage, full.names = TRUE)) {
            target <- file.path(dest, basename(f))
            unlink(target)                  # drops the NAME; a mapped inode lives on
            if (!file.rename(f, target)) file.copy(f, target, overwrite = TRUE)
        }
        unlink(stage, recursive = TRUE)
        TRUE
    }, error = function(e) {
        message(sprintf("sTiles: release download failed (%s)", conditionMessage(e)))
        FALSE
    })
    if (ok && file.exists(lib)) {
        # Superseded solvers are ~20 MB each and serve nobody once a newer one
        # loads; drop them, including any pre-versioning copy.
        old <- setdiff(Sys.glob(file.path(.sTiles_cache_dir(), "*")),
                       file.path(.sTiles_cache_dir(), tag))
        unlink(old[basename(old) == ci | grepl("^v?[0-9]", basename(old))], recursive = TRUE)
        return(lib)
    }
    have <- .sTiles_cached_libs(ci, fname)          # download failed: fall back
    if (length(have)) sort(have, decreasing = TRUE)[1] else NA_character_
}

# Drop the loaded solver so a freshly downloaded one can replace it without
# restarting R. Safe ONLY when this session never built a handle: an "sTiles"
# object carries a C finalizer that lives inside the glue DLL, so unloading
# while one is still reachable would send the next garbage collection into
# code that is no longer mapped. Returns TRUE when nothing is loaded any more.
.sTiles_unload <- function() {
    if (is.null(.sTiles$dll)) return(TRUE)             # never loaded: nothing to do
    if (isTRUE(.sTiles$created > 0L)) return(FALSE)    # finalizers outstanding
    # The DLL table is keyed by the exact string passed to dyn.load, so take
    # the path back from R rather than rebuilding it and missing by a slash.
    dlls <- getLoadedDLLs()
    gluepath <- if (!is.null(dlls[[.sTiles$pkgname]])) dlls[[.sTiles$pkgname]][["path"]]
                else file.path(.sTiles$libname, .sTiles$pkgname, "libs",
                               .Platform$r_arch,
                               paste0(.sTiles$pkgname, .Platform$dynlib.ext))
    ok <- tryCatch({
        dyn.unload(gluepath)          # glue first: its symbols bind into libstiles
        dyn.unload(.sTiles$libpath)
        TRUE
    }, error = function(e) FALSE)
    if (ok) {
        .sTiles$dll <- NULL
        .sTiles$libpath <- NULL
        .sTiles$sym <- new.env(parent = emptyenv())    # cached symbol addresses are stale
    }
    ok
}

#' Delete every cached sTiles solver library
#'
#' Removes the solver copies that [sTiles_install_library()] placed in the
#' package cache. The blunt instrument for a cache believed to be broken or
#' stale, and the way to make room before installing a different release: call
#' [sTiles_install_library()] afterwards to fetch a fresh one. Takes effect
#' immediately unless this session has already built a matrix with the old
#' solver, in which case the files go but a replacement can only load after
#' restarting R (a loaded solver with live handles cannot be swapped
#' underneath them).
#'
#' @return The number of cached solvers removed, invisibly.
#' @seealso [sTiles_install_library()], [sTiles_available()]
#' @examples
#' \dontrun{
#' sTiles_clean_cache()
#' }
#' @export
sTiles_clean_cache <- function() {
    if (.Platform$OS.type == "windows" && !is.null(.sTiles$dll))
        stop("restart R first, then call sTiles_clean_cache(): Windows cannot ",
             "delete a solver that is already loaded", call. = FALSE)
    hits <- .sTiles_cached_libs(.sTiles_ci_folder(), .sTiles_lib_filename())
    for (h in hits) {
        unlink(dirname(h), recursive = TRUE)
        parent <- dirname(dirname(h))           # the per-release directory
        if (!length(list.files(parent, all.files = TRUE, no.. = TRUE)))
            unlink(parent, recursive = TRUE)
    }
    unloaded <- .sTiles_unload()
    message(sprintf("sTiles: removed %d cached solver(s)", length(hits)))
    if (length(hits) && !unloaded)
        message("sTiles: restart R before reinstalling (this session is still ",
                "using the solver just removed).")
    invisible(length(hits))
}

# Every local place a solver could be, best first. Pure path arithmetic plus
# file.exists(): no network, no loading, no side effects.
.sTiles_lib_candidates <- function(libname = NULL, pkgname = NULL) {
    fname <- .sTiles_lib_filename()
    ci <- .sTiles_ci_folder()
    cands <- character(0)

    env_lib <- Sys.getenv("STILES_LIB", "")
    if (nzchar(env_lib)) cands <- c(cands, env_lib)

    env_dir <- Sys.getenv("STILES_LIB_DIR", "")
    if (nzchar(env_dir)) cands <- c(cands, file.path(env_dir, fname))

    # CI binaries tree: <root>/libstiles-<ci>/lib/libstiles.{so,dylib}.
    env_bin <- Sys.getenv("STILES_BINARIES_DIR", "")
    if (nzchar(env_bin)) cands <- c(cands, file.path(env_bin, ci, "lib", fname))

    pkgdir <- if (!is.null(libname) && !is.null(pkgname))
        file.path(libname, pkgname) else system.file(package = "sTiles")

    # Search a `binaries/` tree above the package AND above the working dir
    # (covers an installed package run from inside the repo checkout).
    cands <- c(cands,
               .sTiles_binaries_candidates(pkgdir, ci, fname),
               .sTiles_binaries_candidates(getwd(), ci, fname))

    # Bundled inside the installed package: inst/solver/<plat>/ becomes
    # <pkg>/solver/<plat>/. NOT inst/libs: R reserves <pkg>/libs for the
    # package's own compiled code, and a non-empty inst/libs is a check
    # warning. The libs/ paths stay in the list for installs made before the
    # rename, which cost nothing to try.
    cands <- c(cands,
               file.path(pkgdir, "solver", .sTiles_platform_tag(), fname),
               file.path(pkgdir, "solver", fname),
               file.path(pkgdir, "libs", .sTiles_platform_tag(), fname),
               file.path(pkgdir, "libs", fname))

    # Development checkout: search ancestors for lib/libstiles.{so,dylib}.
    here <- pkgdir
    for (i in seq_len(10)) {
        cands <- c(cands, file.path(here, "lib", fname))
        parent <- dirname(here)
        if (parent == here) break
        here <- parent
    }

    # Last: the cache sTiles_install_library() fills, newest release first.
    # Reached by path alone, so an installed solver is found offline and
    # without asking the releases API which tag is current.
    for (cand in .sTiles_ci_candidates())
        cands <- c(cands, sort(.sTiles_cached_libs(cand, fname), decreasing = TRUE))

    cands
}

#' Is the sTiles solver library available?
#'
#' Reports whether a solver library can be found on this machine, checking the
#' `STILES_LIB`, `STILES_LIB_DIR` and `STILES_BINARIES_DIR` environment
#' variables, a copy bundled in the installed package, a development checkout,
#' and the cache filled by [sTiles_install_library()]. It looks at the file
#' system only: it never contacts the network and never loads anything, which
#' makes it the right guard for examples and tests that have to be skipped
#' where no solver is installed.
#'
#' @return `TRUE` when a solver library was found, `FALSE` otherwise.
#' @seealso [sTiles_install_library()]
#' @examples
#' sTiles_available()
#' @export
sTiles_available <- function()
    any(file.exists(.sTiles_lib_candidates(.sTiles$libname, .sTiles$pkgname)))

#' Install the sTiles solver library
#'
#' Downloads the prebuilt solver library that matches this platform from the
#' sTiles project's releases and stores it in the package cache, under
#' `tools::R_user_dir("sTiles", "cache")`. Run it once: every later session
#' finds the cached copy without touching the network.
#'
#' The solver is a separate component under its own license terms (see the
#' NOTICE file in this package) and is deliberately not bundled. Nothing is
#' downloaded unless you call this function or accept the prompt an interactive
#' session shows the first time a solver is needed. To skip the download
#' altogether, point `STILES_LIB` at a shared object you already have, or
#' `STILES_LIB_DIR` at the directory holding it. Setting `STILES_NO_DOWNLOAD`
#' disables the download path entirely.
#'
#' @section Search order:
#' Before anything is downloaded, and on every later call, the package takes
#' the first solver it finds among:
#' \enumerate{
#'   \item `STILES_LIB`, a shared object named outright;
#'   \item `STILES_LIB_DIR`, a directory holding one;
#'   \item `STILES_BINARIES_DIR`, a CI-artifact tree;
#'   \item a `binaries/` tree above the installed package or the working
#'     directory;
#'   \item a copy bundled in the installed package, under `solver/`;
#'   \item `lib/` in a development checkout above the package;
#'   \item the download cache, newest release first.
#' }
#'
#' @param tag Release tag to install, for example "v2026.8.27". Defaults to the
#'   latest release.
#' @param variant Build variant to prefer, for example "armv82-armpl" or
#'   "v3-mkl", or "none" for the portable default. Defaults to the best fit for
#'   this CPU, falling back to the portable build when the release does not
#'   carry the preferred one.
#' @param force Re-download even when a matching solver is already cached.
#' @return The path of the installed shared library, invisibly.
#' @seealso [sTiles_available()], [sTiles_clean_cache()]
#' @examples
#' \dontrun{
#' sTiles_install_library()
#' sTiles_install_library(tag = "v2026.8.27", variant = "none")
#' }
#' @export
sTiles_install_library <- function(tag = NULL, variant = NULL, force = FALSE) {
    if (nzchar(Sys.getenv("STILES_NO_DOWNLOAD", "")))
        stop("STILES_NO_DOWNLOAD is set, so the solver download is disabled. ",
             "Unset it, or point STILES_LIB at a copy you already have.",
             call. = FALSE)

    keys <- c("STILES_RELEASE_TAG", "STILES_VARIANT")
    old <- Sys.getenv(keys, names = TRUE, unset = NA)
    on.exit({
        for (k in keys)
            if (is.na(old[[k]])) Sys.unsetenv(k)
            else do.call(Sys.setenv, structure(list(old[[k]]), names = k))
    }, add = TRUE)
    if (!is.null(tag)) Sys.setenv(STILES_RELEASE_TAG = tag)
    if (!is.null(variant)) Sys.setenv(STILES_VARIANT = variant)

    got <- .sTiles_download_from_release(force = isTRUE(force))
    if (is.na(got) || !file.exists(got))
        stop("could not install the sTiles solver library. Check the network ",
             "connection, or download the shared library for this platform ",
             "from ", .sTiles_release_page(), " by hand and point STILES_LIB ",
             "at it.", call. = FALSE)
    got <- normalizePath(got)
    message("sTiles: solver installed at ", got)
    invisible(got)
}

# Where a user goes to fetch the library by hand.
.sTiles_release_page <- function()
    sprintf("https://github.com/%s/releases",
            Sys.getenv("STILES_RELEASE_REPO", "esmail-abdulfattah/sTiles"))

.sTiles_find_lib <- function(libname, pkgname) {
    fname <- .sTiles_lib_filename()
    ci <- .sTiles_ci_folder()
    cands <- .sTiles_lib_candidates(libname, pkgname)

    hit <- cands[file.exists(cands)]
    if (length(hit) > 0) return(normalizePath(hit[1]))

    # Nothing local. Ask when there is somebody to ask; never download on our
    # own initiative, and in a script say plainly what to run instead.
    if (interactive() && !nzchar(Sys.getenv("STILES_NO_DOWNLOAD", ""))) {
        ans <- tryCatch(utils::askYesNo(
            sprintf(paste0("sTiles: the solver library (%s) is not installed.\n",
                           "Download it now from the sTiles releases?"), fname),
            default = FALSE), error = function(e) FALSE)
        if (isTRUE(ans)) {
            dl <- .sTiles_download_from_release()
            if (!is.na(dl) && file.exists(dl)) return(normalizePath(dl))
        }
    }

    stop("the sTiles solver library (", fname, ") is not installed. Run ",
         "sTiles_install_library() once to download it, or set STILES_LIB to ",
         "a copy you already have. Searched ", length(cands), " locations; ",
         "see ?sTiles_install_library for the search order.", call. = FALSE)
}

.onLoad <- function(libname, pkgname) {
    # Quiet libstiles' one-time banner unless the user opted in.
    if (Sys.getenv("STILES_NO_BANNER", "") == "")
        Sys.setenv(STILES_NO_BANNER = "1")

    # Defer locating/loading libstiles + the glue to first use, so that
    # install (R CMD INSTALL test-load) and library(sTiles) never fail merely
    # because the binary hasn't been downloaded from the Release yet. The
    # download happens on the first actual sTiles call (see .sTiles_ensure_loaded).
    .sTiles$libname <- libname
    .sTiles$pkgname <- pkgname
    .sTiles$sym <- new.env(parent = emptyenv())
    .sTiles$created <- 0L   # handles built here; gates .sTiles_unload()
}

# Locate/download libstiles and load it + the glue DLL. Idempotent; called from
# every native entry point via .sc().
.sTiles_ensure_loaded <- function() {
    if (!is.null(.sTiles$dll)) return(invisible())
    .sTiles_check_cpu()
    libpath <- .sTiles_find_lib(.sTiles$libname, .sTiles$pkgname)
    # Preload libstiles with GLOBAL symbol visibility (local = FALSE) so the
    # glue's undefined sTiles_* symbols resolve against it. (Windows has no
    # such mechanism; see the GetProcAddress bind step below instead.)
    dyn.load(libpath, local = FALSE, now = TRUE)
    .sTiles$libpath <- libpath

    gluepath <- file.path(.sTiles$libname, .sTiles$pkgname, "libs",
                          .Platform$r_arch,
                          paste0(.sTiles$pkgname, .Platform$dynlib.ext))
    .sTiles$dll <- dyn.load(gluepath)

    # Windows PE/DLL linking can't leave the glue's sTiles_* symbols undefined
    # at link time the way an ELF .so (Linux) or a -undefined dynamic_lookup
    # .dylib (macOS) can, so the glue resolves them itself via
    # LoadLibrary/GetProcAddress once it knows the real libstiles.dll path.
    if (.Platform$OS.type == "windows")
        .Call(.sTiles_win_bind_sym(), libpath)

    invisible()
}

# Address of the Windows bind routine, or its NAME before anything is loaded.
# See .sc for why the name, and not an error, is the right answer there.
.sTiles_win_bind_sym <- function() {
    if (is.null(.sTiles$dll)) return("sTiles_win_bind_R")
    getNativeSymbolInfo("sTiles_win_bind_R", PACKAGE = .sTiles$dll)$address
}

.onUnload <- function(libpath) {
    if (!is.null(.sTiles$dll)) try(dyn.unload(.sTiles$dll[["path"]]), silent = TRUE)
}

# Resolve (and cache) a registered native routine from the glue DLL, or return
# the routine's NAME when nothing is loaded yet.
#
# Loading is NOT done here: every entry point calls .sTiles_ensure_loaded()
# before it reaches a .Call, so a real call always has the address by now, and
# a missing solver is reported there with a message saying what to install.
# The name matters for R CMD check, which evaluates this expression on its own,
# on a machine that has no solver: an error raised here is reported as a
# registration problem, which is what CRAN's incoming check rejects. A
# character is a legal first argument to .Call and the routines are registered
# (R_registerRoutines in sTiles_glue.c), so the fallback still resolves rather
# than merely quietening a check.
.sc <- function(name) {
    if (is.null(.sTiles$dll)) return(name)
    s <- .sTiles$sym[[name]]
    if (is.null(s)) {
        s <- getNativeSymbolInfo(name, PACKAGE = .sTiles$dll)$address
        assign(name, s, envir = .sTiles$sym)
    }
    s
}

#' Path of the loaded sTiles solver library
#'
#' Loads the solver if this session has not loaded it yet, then reports which
#' file answered. Useful when several builds are installed and you need to know
#' which one is in use.
#'
#' @return The absolute path of the loaded shared library, as a string.
#' @seealso [sTiles_available()], [sTiles_install_library()]
#' @examples
#' if (sTiles_available()) {
#'   sTiles_library_path()
#' }
#' @export
sTiles_library_path <- function() { .sTiles_ensure_loaded(); .sTiles$libpath }

#' Version of the sTiles solver library
#'
#' @return The solver's version string.
#' @seealso [sTiles_summary()]
#' @examples
#' if (sTiles_available()) {
#'   sTiles_version()
#' }
#' @export
sTiles_version <- function() {
    .sTiles_ensure_loaded()
    .Call(.sc("sTiles_version_R"))
}

# ---------------------------------------------------------------------------
# Matrix -> lower-triangle COO (0-based, canonical (row, col) order).
# ---------------------------------------------------------------------------
.sTiles_lower_coo <- function(Q) {
    if (is.matrix(Q)) Q <- methods::as(Q, "CsparseMatrix")
    if (!methods::is(Q, "sparseMatrix"))
        Q <- methods::as(methods::as(Q, "matrix"), "CsparseMatrix")
    if (nrow(Q) != ncol(Q)) stop("matrix must be square")
    n <- nrow(Q)

    L <- Matrix::tril(methods::as(Q, "CsparseMatrix"))
    L <- methods::as(L, "TsparseMatrix")   # triplet form: @i, @j, @x (0-based)
    i <- L@i; j <- L@j; x <- L@x

    ord <- order(i, j)                     # canonical, stable across refactors
    list(n = n, i = as.integer(i[ord]), j = as.integer(j[ord]),
         x = as.double(x[ord]))
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#' Preprocess a matrix: ordering bake-off and tile layout only
#'
#' The symbolic phase. It depends only on the sparsity pattern, not on the
#' numeric values, and performs no Cholesky. Follow it with [sTiles_factorize()]
#' for the numeric factorization. Timing the two apart separates the
#' preprocessing cost, paid once per pattern, from the numeric cost, paid once
#' per set of values.
#'
#' @inheritParams sTiles
#' @return An object of class "sTiles", analyzed but not yet factorized.
#' @seealso [sTiles_factorize()], [sTiles_update()], [sTiles()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles_analyze(Q)
#'   sTiles_factorize(s)
#'   sTiles_logdet(s)
#'   sTiles_close(s)
#' }
#' @export
sTiles_analyze <- function(Q, cores = 1L, mode = "auto", tile_size = 40L,
                           inverse = FALSE, log_level = -1L) {
    m <- if (is.character(mode)) {
        code <- .sTiles_modes[tolower(mode)]
        if (is.na(code)) stop("unknown mode '", mode, "'")
        code
    } else as.integer(mode)

    .sTiles_ensure_loaded()
    coo <- .sTiles_lower_coo(Q)
    ## Timed here: libstiles reports chol/selinv time but nothing for the
    ## preprocessing, which is usually the expensive phase -- it runs once per
    ## sparsity pattern, while sTiles_factorize() runs once per set of values.
    t0 <- proc.time()[["elapsed"]]
    ptr <- .Call(.sc("sTiles_analyze_R"), coo$i, coo$j, coo$n,
                 as.integer(cores), as.integer(m), as.integer(tile_size),
                 as.logical(inverse), 0L, as.integer(log_level))
    analyze_time <- proc.time()[["elapsed"]] - t0

    .sTiles$created <- .sTiles$created + 1L   # a C finalizer now exists; see .sTiles_unload

    obj <- list(ptr = ptr, n = coo$n, nnz = length(coo$i), analyze_time = analyze_time,
                mode = as.integer(m), cores = as.integer(cores),
                inverse = isTRUE(inverse), factored = FALSE,
                values = coo$x, pattern = list(i = coo$i, j = coo$j))
    class(obj) <- "sTiles"
    obj
}

#' Numeric Cholesky factorization, reusing the preprocessing
#'
#' Runs the numeric phase on an object that [sTiles_analyze()] has already
#' prepared, so the ordering and tile layout are not recomputed.
#'
#' @param x An "sTiles" object from [sTiles_analyze()] or [sTiles()].
#' @param Q Optional: a matrix with the SAME sparsity pattern, whose values to
#'   factor. When omitted, the values captured at analyze time are used.
#' @return The "sTiles" object, now factorized, invisibly.
#' @seealso [sTiles_analyze()], [sTiles_update()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles_analyze(Q)
#'   sTiles_factorize(s)
#'   sTiles_logdet(s)
#'   sTiles_close(s)
#' }
#' @export
sTiles_factorize <- function(x, Q = NULL) {
    .sTiles_ensure_loaded()
    vals <- if (is.null(Q)) x$values else {
        coo <- .sTiles_lower_coo(Q)
        if (length(coo$i) != x$nnz ||
            !identical(coo$i, x$pattern$i) || !identical(coo$j, x$pattern$j))
            stop("sTiles_factorize(Q=): Q must share this object's sparsity ",
                 "pattern; build a new sTiles_analyze() otherwise", call. = FALSE)
        coo$x
    }
    .Call(.sc("sTiles_factorize_R"), x$ptr, as.double(vals))
    invisible(x)
}

#' New values, same sparsity pattern: refactorize without reanalyzing
#'
#' The ordering and tile layout depend only on WHERE the non-zeros are, so an
#' object built by [sTiles_analyze()] can absorb any number of value updates
#' and pay only the numeric cost each time. This is the loop an iterative
#' method wants.
#'
#' @param x An "sTiles" object from [sTiles_analyze()].
#' @param Q A matrix with the SAME sparsity pattern, whose values to factor.
#' @return The "sTiles" object, factorized with the new values, invisibly.
#' @seealso [sTiles_analyze()], [sTiles_factorize()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles_analyze(Q)
#'   sTiles_factorize(s)
#'   d1 <- sTiles_logdet(s)
#'   ## same pattern, different values: no new ordering
#'   sTiles_update(s, Q * 2)
#'   d2 <- sTiles_logdet(s)
#'   c(d1, d2)
#'   sTiles_close(s)
#' }
#' @export
sTiles_update <- function(x, Q) {
    if (missing(Q) || is.null(Q))
        stop("sTiles_update(): supply the matrix whose values to use", call. = FALSE)
    sTiles_factorize(x, Q)
}

#' Factorize a symmetric positive-definite matrix
#'
#' One shot: runs the preprocessing ([sTiles_analyze()]) and then the numeric
#' factorization ([sTiles_factorize()]). To time the two phases apart, or to
#' reuse one preprocessing across many sets of values, call those two yourself.
#'
#' @param Q A symmetric positive-definite matrix, either a
#'   `Matrix::sparseMatrix` or a base matrix. Only the lower triangle is read.
#' @param cores Worker threads (default 1).
#' @param mode Tile regime: "auto" (default), "dense", "semisparse" or
#'   "sparse".
#' @param tile_size Tile size, or -1 to let the solver choose (default 40).
#' @param inverse Reserve selected-inverse storage, required by
#'   [sTiles_selinv()] and the `sTiles_selinv_*()` queries (default `FALSE`).
#' @param log_level Solver verbosity: -1 silent (default), 0 timing, 1 info,
#'   2 debug, 3 trace.
#' @return An object of class "sTiles" wrapping a live factorization.
#' @seealso [sTiles_logdet()], [sTiles_solve()], [sTiles_selinv()],
#'   [sTiles_summary()], [sTiles_close()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q)
#'   sTiles_logdet(s)
#'   sTiles_solve(s, rep(1, 50))
#'   sTiles_close(s)
#' }
#' @export
sTiles <- function(Q, cores = 1L, mode = "auto", tile_size = 40L,
                   inverse = FALSE, log_level = -1L) {
    s <- sTiles_analyze(Q, cores = cores, mode = mode, tile_size = tile_size,
                        inverse = inverse, log_level = log_level)
    sTiles_factorize(s)
    s
}

#' Log-determinant of the factorized matrix
#'
#' Returns `log(det(Q))`, computed from the factor as `2 * sum(log(diag(L)))`.
#'
#' @param x A factorized "sTiles" object.
#' @return The log-determinant, a single number.
#' @seealso [sTiles()], [sTiles_summary()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q)
#'   sTiles_logdet(s)
#'   sTiles_close(s)
#' }
#' @export
sTiles_logdet <- function(x) {
    .sTiles_ensure_loaded()
    .Call(.sc("sTiles_logdet_R"), x$ptr)
}

#' Compute the selected inverse, reusing the current factorization
#'
#' Computes Z, the inverse of Q restricted to the pattern of the Cholesky
#' factor, `pattern(L + L^T)`. The object must have been built with
#' `inverse = TRUE`. The call is idempotent, and the computation otherwise
#' happens lazily on the first `sTiles_selinv_*()` query. Call it explicitly to
#' time the selected inverse on its own, and call it again after each
#' [sTiles_factorize()] to refresh Z for the new values.
#'
#' @param x A factorized "sTiles" object built with `inverse = TRUE`.
#' @return The "sTiles" object, invisibly.
#' @seealso [sTiles_selinv_diag()], [sTiles_selinv_elm()],
#'   [sTiles_selinv_row()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q, inverse = TRUE)
#'   sTiles_selinv(s)
#'   head(sTiles_selinv_diag(s))
#'   sTiles_close(s)
#' }
#' @export
sTiles_selinv <- function(x) {
    .sTiles_ensure_loaded()
    .Call(.sc("sTiles_selinv_R"), x$ptr)
    invisible(x)
}

#' Diagonal of the selected inverse: the marginal variances
#'
#' Returns `diag(solve(Q))`, in the original ordering. Triggers the
#' selected-inverse computation on first use.
#'
#' @param x A factorized "sTiles" object built with `inverse = TRUE`.
#' @return A numeric vector of length `n`, the diagonal of the inverse.
#' @seealso [sTiles_selinv()], [sTiles_selinv_elm()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q, inverse = TRUE)
#'   head(sTiles_selinv_diag(s))
#'   sTiles_close(s)
#' }
#' @export
sTiles_selinv_diag <- function(x) {
    .sTiles_ensure_loaded()
    .Call(.sc("sTiles_selinv_diag_R"), x$ptr)
}

#' One entry of the selected inverse
#'
#' Returns the selected inverse at position `(i, j)` when that position lies in
#' the factor pattern, `pattern(L + L^T)`, and exactly 0 outside it. Both
#' triangles are accepted, since Z is symmetric. Triggers the selected-inverse
#' computation on first use.
#'
#' @param x A factorized "sTiles" object built with `inverse = TRUE`.
#' @param i,j Row and column, 1-based, in the original ordering.
#' @return The entry as a single number, 0 outside the factor pattern.
#' @seealso [sTiles_selinv()], [sTiles_selinv_row()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q, inverse = TRUE)
#'   sTiles_selinv_elm(s, 1, 2)
#'   sTiles_close(s)
#' }
#' @export
sTiles_selinv_elm <- function(x, i, j) {
    .sTiles_ensure_loaded()
    .Call(.sc("sTiles_selinv_elm_R"), x$ptr, as.integer(i), as.integer(j))
}

#' Several entries from one row of the selected inverse
#'
#' Returns the selected inverse at `(node, k)` for each `k` in `neighbors`, the
#' access pattern a graph model wants. Entries outside the factor pattern come
#' back as 0.
#'
#' @param x A factorized "sTiles" object built with `inverse = TRUE`.
#' @param node Row index, 1-based, in the original ordering.
#' @param neighbors Integer vector of column indices, 1-based.
#' @return A numeric vector, one value per entry of `neighbors`.
#' @seealso [sTiles_selinv()], [sTiles_selinv_elm()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q, inverse = TRUE)
#'   sTiles_selinv_row(s, 5, c(4, 5, 6))
#'   sTiles_close(s)
#' }
#' @export
sTiles_selinv_row <- function(x, node, neighbors) {
    .sTiles_ensure_loaded()
    .Call(.sc("sTiles_selinv_row_R"), x$ptr, as.integer(node),
          as.integer(neighbors))
}

#' Solve a linear system with the factorization
#'
#' @param x A factorized "sTiles" object.
#' @param b Right-hand side: a length-`n` vector, or an `n` by `nrhs` matrix.
#' @param system Which system to solve: "A" for `Q x = b` (default), "L" for
#'   the forward solve `L y = b`, "Lt" for the backward solve `t(L) x = b`.
#' @return The solution, with the same shape as `b`.
#' @seealso [sTiles()], [sTiles_logdet()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q)
#'   x <- sTiles_solve(s, rep(1, 50))
#'   max(abs(as.vector(Q %*% x) - 1))
#'   sTiles_close(s)
#' }
#' @export
sTiles_solve <- function(x, b, system = c("A", "L", "Lt")) {
    .sTiles_ensure_loaded()
    which <- switch(match.arg(system), A = 0L, L = 1L, Lt = 2L)
    .Call(.sc("sTiles_solve_R"), x$ptr, as.double(b), which)
}

#' Structured summary of a factorization
#'
#' Collects the dimensions, the fill, the tile mode, the phase the object is
#' in, and the timings the solver measured. The result is an ordinary list, so
#' single numbers are easy to pull out (`sTiles_summary(s)$chol_time`), and it
#' prints as a short report.
#'
#' @param x An "sTiles" object.
#' @return An object of class "sTiles_summary": a list with elements `n`,
#'   `nnz`, `nnz_factor`, `mode`, `cores`, `inverse`, `factored`,
#'   `analyze_time`, `chol_time`, `selinv_time`, `version` and `library`.
#' @seealso [sTiles()], [sTiles_version()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q)
#'   sTiles_summary(s)$nnz_factor
#'   sTiles_close(s)
#' }
#' @export
sTiles_summary <- function(x) {
    .sTiles_ensure_loaded()
    modes <- c("dense", "semisparse", "sparse", "auto")
    fac <- as.logical(.Call(.sc("sTiles_is_factored_R"), x$ptr))
    out <- list(
        n           = x$n,
        nnz         = x$nnz,
        nnz_factor  = .Call(.sc("sTiles_nnz_factor_R"), x$ptr),
        mode        = modes[x$mode + 1L],
        cores       = x$cores,
        inverse     = x$inverse,
        factored    = fac,
        analyze_time = if (!is.null(x$analyze_time)) x$analyze_time else NA_real_,
        chol_time   = if (fac) .Call(.sc("sTiles_chol_time_R"), x$ptr) else NA_real_,
        selinv_time = tryCatch(.Call(.sc("sTiles_selinv_time_R"), x$ptr),
                               error = function(e) NA_real_),
        version     = sTiles_version(),
        library     = .sTiles$libpath)
    class(out) <- "sTiles_summary"
    out
}

#' Free a factorization now
#'
#' Releases the solver's memory immediately. Optional: an object that goes out
#' of scope is freed at the next garbage collection anyway. Worth calling in a
#' loop over large matrices, where waiting for the collector means holding
#' several factorizations at once.
#'
#' @param x An "sTiles" object.
#' @return `NULL`, invisibly.
#' @seealso [sTiles()]
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   s <- sTiles(Q)
#'   sTiles_close(s)
#' }
#' @export
sTiles_close <- function(x) {
    .sTiles_ensure_loaded()
    invisible(.Call(.sc("sTiles_free_R"), x$ptr))
}

#' Print a sTiles factorization object
#'
#' @param x An "sTiles" object.
#' @param ... Ignored, present for consistency with [print()].
#' @return `x`, invisibly.
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   print(sTiles(Q))
#' }
#' @export
print.sTiles <- function(x, ...) {
    .sTiles_ensure_loaded()
    modes <- c("dense", "semisparse", "sparse", "auto")
    fac <- as.logical(.Call(.sc("sTiles_is_factored_R"), x$ptr))
    cat(sprintf("<sTiles: %d x %d, nnz=%d, mode=%s, cores=%d, inverse=%s, %s>\n",
                x$n, x$n, x$nnz, modes[x$mode + 1L], x$cores, x$inverse,
                if (fac) "factorized" else "analyzed"))
    invisible(x)
}

#' Print a sTiles summary
#'
#' @param x An object of class "sTiles_summary" from [sTiles_summary()].
#' @param ... Ignored, present for consistency with [print()].
#' @return `x`, invisibly.
#' @examples
#' if (sTiles_available()) {
#' Q <- Matrix::bandSparse(50, k = c(0, 1),
#'                         diagonals = list(rep(4, 50), rep(-1, 49)),
#'                         symmetric = TRUE)
#'   print(sTiles_summary(sTiles(Q)))
#' }
#' @export
print.sTiles_summary <- function(x, ...) {
    fmt_t <- function(t) if (is.na(t)) "-" else sprintf("%.4g s", t)
    cat(sprintf(paste0(
        "sTiles factorization\n",
        "  dimension  : %d x %d\n",
        "  input nnz  : %d      factor nnz(L) : %d\n",
        "  tile mode  : %s   cores : %d   inverse : %s\n",
        "  state      : %s\n",
        "  analyze    : %s   (once per sparsity pattern)\n",
        "  chol time  : %s      selinv time : %s\n",
        "  library    : %s\n"),
        x$n, x$n, x$nnz, x$nnz_factor, x$mode, x$cores, x$inverse,
        if (x$factored) "factorized" else "analyzed (not factorized)",
        fmt_t(x$analyze_time), fmt_t(x$chol_time), fmt_t(x$selinv_time), x$library))
    invisible(x)
}
