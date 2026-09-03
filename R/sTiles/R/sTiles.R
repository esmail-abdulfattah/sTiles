## sTiles -- R interface to the sTiles sparse Cholesky / selected-inverse
## framework.  See ?sTiles for the entry point.

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
    if (Sys.info()[["sysname"]] == "Darwin") "libstiles.dylib" else "libstiles.so"
}

.sTiles_platform_tag <- function() {
    sysname <- Sys.info()[["sysname"]]
    machine <- Sys.info()[["machine"]]
    arch <- switch(machine,
                   "x86_64" = "x86_64", "amd64" = "x86_64",
                   "arm64" = "arm64", "aarch64" = "arm64", machine)
    os <- if (sysname == "Darwin") "macos" else "linux"
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

# ---------------------------------------------------------------------------
# Fetch the matching prebuilt libstiles from the GitHub Release.
#
# When the package is install_github()'d there is no binary in the tree, so on
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

# first use we download the platform library from the project's Release assets
# and cache it. The Linux/macOS builds are self-contained (BLAS embedded).
# Overrides: STILES_NO_DOWNLOAD, STILES_RELEASE_REPO, STILES_RELEASE_BASE_URL,
# STILES_CACHE_DIR.
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

#' Delete every cached solver, so the next call downloads a fresh one.
#'
#' The blunt instrument for a cache believed to be broken or stale, or the way
#' to pick up a new release without waiting for a version bump: the next
#' sTiles call re-downloads the solver from scratch. Takes effect immediately
#' unless this session has already built a matrix with the old solver, in
#' which case the fetch happens but loading it waits for a restart (a loaded
#' solver with live handles cannot be replaced underneath them).
#'
#' @return Number of cached solvers removed, invisibly.
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
        message("sTiles: restart R for a fresh download (this session is still ",
                "using the solver just removed).")
    invisible(length(hits))
}

.sTiles_find_lib <- function(libname, pkgname) {
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

    pkgdir <- if (!missing(libname) && !missing(pkgname))
        file.path(libname, pkgname) else system.file(package = "sTiles")

    # Search a `binaries/` tree above the package AND above the working dir
    # (covers an installed package run from inside the repo checkout).
    cands <- c(cands,
               .sTiles_binaries_candidates(pkgdir, ci, fname),
               .sTiles_binaries_candidates(getwd(), ci, fname))

    # Bundled inside the installed package: inst/libs/<plat>/ -> libs/<plat>/.
    cands <- c(cands,
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

    hit <- cands[file.exists(cands)]
    if (length(hit) > 0) return(normalizePath(hit[1]))

    # Nothing local: fetch the prebuilt library from the GitHub Release.
    dl <- .sTiles_download_from_release()
    if (!is.na(dl) && file.exists(dl)) return(normalizePath(dl))

    stop("Could not locate ", fname, ".\nThe automatic download from the GitHub ",
         "Release failed or was disabled. Set STILES_LIB to the shared object, ",
         "STILES_LIB_DIR to its directory, or STILES_BINARIES_DIR to a CI-artifact ",
         "tree (", ci, "/lib/", fname, ").\nSearched:\n  ",
         paste(cands, collapse = "\n  "), call. = FALSE)
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
        .Call(getNativeSymbolInfo("sTiles_win_bind_R", PACKAGE = .sTiles$dll)$address,
              libpath)

    invisible()
}

.onUnload <- function(libpath) {
    if (!is.null(.sTiles$dll)) try(dyn.unload(.sTiles$dll[["path"]]), silent = TRUE)
}

# Resolve (and cache) a registered native routine from the glue DLL.
.sc <- function(name) {
    .sTiles_ensure_loaded()
    s <- .sTiles$sym[[name]]
    if (is.null(s)) {
        s <- getNativeSymbolInfo(name, PACKAGE = .sTiles$dll)$address
        assign(name, s, envir = .sTiles$sym)
    }
    s
}

#' Absolute path of the loaded libstiles shared object.
#' @export
sTiles_library_path <- function() { .sTiles_ensure_loaded(); .sTiles$libpath }

#' sTiles library version string.
#' @export
sTiles_version <- function() .Call(.sc("sTiles_version_R"))

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

#' Preprocess (analyze) a matrix: ordering bake-off + tile layout ONLY.
#'
#' This is the symbolic phase -- it depends only on the sparsity pattern, not
#' the numeric values, and does no Cholesky. Follow it with sTiles_factorize()
#' to run the numeric factorization; timing the two separately isolates the
#' preprocessing cost from the numeric cost.
#'
#' @inheritParams sTiles
#' @return An object of class "sTiles" that is analyzed but not yet factorized.
#' @export
sTiles_analyze <- function(Q, cores = 1L, mode = "auto", tile_size = 40L,
                           inverse = FALSE, log_level = -1L) {
    m <- if (is.character(mode)) {
        code <- .sTiles_modes[tolower(mode)]
        if (is.na(code)) stop("unknown mode '", mode, "'")
        code
    } else as.integer(mode)

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

#' Numeric Cholesky factorization (reuses the preprocessing from analyze).
#'
#' @param x  An "sTiles" object from sTiles_analyze() (or sTiles()).
#' @param Q  Optional: a matrix with the SAME sparsity pattern whose values to
#'   factor. If omitted, the values captured at analyze time are used.
#' @return The (invisibly returned) "sTiles" object, now factorized.
#' @export
sTiles_factorize <- function(x, Q = NULL) {
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

#' New values, same sparsity pattern: re-factorize without re-analyzing.
#'
#' The ordering and tile layout depend only on WHERE the non-zeros are, so an
#' object built by sTiles_analyze() can absorb any number of value updates and
#' pay only the numeric cost each time. This is the loop an iterative method
#' wants.
#'
#' @param x  An "sTiles" object from sTiles_analyze().
#' @param Q  A matrix with the SAME sparsity pattern, whose values to factor.
#' @return The (invisibly returned) "sTiles" object, factorized with the new
#'   values.
#' @export
sTiles_update <- function(x, Q) {
    if (missing(Q) || is.null(Q))
        stop("sTiles_update(): supply the matrix whose values to use", call. = FALSE)
    sTiles_factorize(x, Q)
}

#' Factorize a symmetric positive-definite matrix with sTiles.
#'
#' One-shot: runs preprocessing (sTiles_analyze) then the numeric factorization
#' (sTiles_factorize). To time the two phases apart, or to reuse preprocessing
#' across many value-sets, call sTiles_analyze() + sTiles_factorize() yourself.
#'
#' @param Q  A symmetric positive-definite matrix (Matrix::sparseMatrix or a
#'   base matrix).  Only the lower triangle is used.
#' @param cores  Worker threads (default 1).
#' @param mode  "auto", "dense", "semisparse" or "sparse" (default "auto").
#' @param tile_size  Tile size, or -1 for auto (default 40).
#' @param inverse  Reserve selected-inverse storage; required for
#'   sTiles_selinv()/_diag()/_elm()/_row() (default FALSE).
#' @param log_level  libstiles verbosity: -1 silent (default), 0 timing,
#'   1 info, 2 debug, 3 trace.
#' @return An object of class "sTiles" wrapping a live factorization.
#' @export
sTiles <- function(Q, cores = 1L, mode = "auto", tile_size = 40L,
                   inverse = FALSE, log_level = -1L) {
    s <- sTiles_analyze(Q, cores = cores, mode = mode, tile_size = tile_size,
                        inverse = inverse, log_level = log_level)
    sTiles_factorize(s)
    s
}

#' Log-determinant of Q ( = 2 * sum(log diag(L)) ).
#' @export
sTiles_logdet <- function(x) .Call(.sc("sTiles_logdet_R"), x$ptr)

#' Compute the selected inverse, reusing the current numeric factorization.
#'
#' Z = Q^-1 restricted to the pattern of the Cholesky factor (pattern(L+L^T)).
#' Requires the handle to have been built with inverse = TRUE. Idempotent, and
#' otherwise computed lazily on the first sTiles_selinv_*() query. Call it
#' explicitly to time the selected inverse on its own, and re-call it after each
#' sTiles_factorize() to refresh Z for new values.
#' @return The (invisibly returned) "sTiles" object.
#' @export
sTiles_selinv <- function(x) {
    .Call(.sc("sTiles_selinv_R"), x$ptr)
    invisible(x)
}

#' Diagonal of the selected inverse, diag(Q^-1) -- the marginal variances.
#' @export
sTiles_selinv_diag <- function(x) .Call(.sc("sTiles_selinv_diag_R"), x$ptr)

#' Selected-inverse entry (Q^-1)[i, j] at ANY position (1-based, original order).
#'
#' Returns the selected inverse at (i, j) when that position lies in the factor
#' pattern (pattern(L+L^T)), and exactly 0 outside it. Both triangles work (Z is
#' symmetric). Triggers the selected-inverse computation on first use.
#' @export
sTiles_selinv_elm <- function(x, i, j)
    .Call(.sc("sTiles_selinv_elm_R"), x$ptr, as.integer(i), as.integer(j))

#' Selected-inverse values (Q^-1)[node, k] for each k in `neighbors` (1-based).
#' @export
sTiles_selinv_row <- function(x, node, neighbors)
    .Call(.sc("sTiles_selinv_row_R"), x$ptr, as.integer(node),
          as.integer(neighbors))

#' Solve with the factorization.
#'
#' @param x  A factorized "sTiles" object.
#' @param b  Right-hand side: a length-n vector or an n x nrhs matrix.
#' @param system  Which system to solve: "A" for Q x = b (default), "L" for
#'   L y = b (forward), "Lt" for L^T x = b (backward).
#' @return The solution, same shape as `b`.
#' @export
sTiles_solve <- function(x, b, system = c("A", "L", "Lt")) {
    which <- switch(match.arg(system), A = 0L, L = 1L, Lt = 2L)
    .Call(.sc("sTiles_solve_R"), x$ptr, as.double(b), which)
}

#' Structured summary of a factorization: dimensions, fill, mode, phase state,
#' and library-measured timings. Returns a list (e.g. sTiles_summary(s)$chol_time)
#' and prints a short report.
#' @export
sTiles_summary <- function(x) {
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

#' Free the factorization now (otherwise freed at garbage collection).
#' @export
sTiles_close <- function(x) invisible(.Call(.sc("sTiles_free_R"), x$ptr))

#' @export
print.sTiles <- function(x, ...) {
    modes <- c("dense", "semisparse", "sparse", "auto")
    fac <- as.logical(.Call(.sc("sTiles_is_factored_R"), x$ptr))
    cat(sprintf("<sTiles: %d x %d, nnz=%d, mode=%s, cores=%d, inverse=%s, %s>\n",
                x$n, x$n, x$nnz, modes[x$mode + 1L], x$cores, x$inverse,
                if (fac) "factorized" else "analyzed"))
    invisible(x)
}

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
